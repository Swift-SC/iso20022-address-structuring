"""
Module providing the `TransformerCRF` class.
"""
from contextlib import suppress
from typing import Sequence

import torch
from torch import nn
from pydantic import BaseModel, Field

from data_structuring.components.details import Details
from data_structuring.components.models.crf_with_marginal import CRFSecondOrder
from data_structuring.components.models.country_head import CountryHead
from data_structuring.components.models.encoder_transformer import EncoderTransformer as BackboneTransformer
from data_structuring.components.models.utils import pos_embed_1d, create_details_from_biotags
from data_structuring.components.tags import BIOTag
from data_structuring.components.types import TorchTensor


class ResultCRF(BaseModel):
    grouped_tags: Details = Field(description="An annotated `Details` object that "
                                              "contains detailed information about the prediction")
    emissions: TorchTensor = Field(description="Emissions produced by the Transformer "
                                               "of shape (SEQUENCE_LENGTH, NUM_TAGS)")
    log_likelihood: TorchTensor = Field(description="Log-likelihood produced by the CRF "
                                                    "of shape (SEQUENCE_LENGTH, NUM_TAGS)")


class TransformerCRF(nn.Module):
    """CRF head with transformer-encoder backbone to perform token-level classification (e.g., NER)."""

    def __init__(
            self,
            vocab_size: int,
            tags: Sequence[BIOTag],
            mapping_id_to_country: dict[int, str],
            max_seq_len: int = 128,
            d_model: int = 64,
            nhead: int = 4,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
            activation: str = "gelu",
            layer_norm_eps: float = 1e-5,
            batch_first: bool = True,
            depth: int = 4,
            padding_idx: int = 0,
            use_country_classifier: bool = True,
            regularisation_emissions: float = 0,
            regularisation_transitions: float = 0,
            regularisation_transitions_order_2: float = 0
    ):
        """Create a TransformerCRF model.

        Args:
            vocab_size (int): The size of the vocabulary.
            tags (Sequence[BIOTag]): All tags to use (i.e., possible entity-level labels).
            max_seq_len (int, optional): The maximum sequence (i.e., sentence) length. Defaults to 128.

            d_model (int, optional): The embedding dimension used in the model. Defaults to 64.
            nhead (int, optional): The number of heads to use in self-attention. Defaults to 4.
            mlp_ratio (float, optional): The number by which `d_model` is
                multiplied to compute the hidden dimension of the MLPs. Defaults to 4.0.
            dropout (float, optional): The dropout rate to use in self-attention. Defaults to 0.1.
            activation (str, optional): The activation function to
                use for the self-attention's MLPs. Defaults to "gelu".
            layer_norm_eps (float, optional): The normalization epsilon
                to use in layer normalization. Defaults to 1e-5.
            batch_first (bool, optional): Whether the inputs are batch-first (`True`),
                or sequence-first (`False`) shaped.
            depth (int, optional): The number of self-attention layers to use. Defaults to 4.
            padding_idx (int, optional): The index of the padding token in the vocabulary. Defaults to 0.

            regularisation_emissions (float, optional): Regularisation for the emissions (Transformer output).
                Defaults to 0.
            regularisation_transitions (float, optional): Regularisation for the order 1 transition matrix.
                Defaults to 0.
            regularisation_transitions_order_2 (float, optional): Regularisation for the order 1 transition matrix.
                Defaults to 0.
        """
        super().__init__()

        # Store arguments
        self._has_country_classifier = use_country_classifier
        self._max_seq_len = max_seq_len
        self._padding_idx = padding_idx
        self._mapping_id_to_country = mapping_id_to_country

        self._idx_to_tag = dict(enumerate(tags))
        self._tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}

        # Embedding lookup table, maps each token to its embedding
        self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # CLS token (used for country prediction)
        self.cls_token = None
        if self._has_country_classifier:
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

        # Fixed cosine positional embeddings for the transformer
        # The "+1" is for the CLS token.
        self.pos_embed = torch.nn.Parameter(
            pos_embed_1d(d_model, (max_seq_len + 1) if self._has_country_classifier else max_seq_len).unsqueeze(0),
            requires_grad=False
        )

        # Transformer encoder layers: Add context to the embeddings through self-attention
        self.transformer_encoder = BackboneTransformer(d_model=d_model,
                                                       nhead=nhead,
                                                       mlp_ratio=mlp_ratio,
                                                       dropout=dropout,
                                                       activation=activation,
                                                       layer_norm_eps=layer_norm_eps,
                                                       batch_first=batch_first,
                                                       depth=depth)

        # Country MLP
        self.country_predictor = None
        if self._has_country_classifier:
            self.country_predictor = CountryHead(embedding_dim=d_model, num_countries=len(mapping_id_to_country))

        # Linear projection from the embedding space to the tags space to obtain logits
        self.projection = torch.nn.Linear(d_model, len(tags))

        # CRF layer on the logits (i.e., emissions)
        # self.crf = HeadCRF(
        #     num_tags=len(tags), batch_first=batch_first, is_order_2=crf_is_order_2
        # )
        self.crf = CRFSecondOrder(num_tags=len(tags), batch_first=batch_first)

        # Loss regularisation
        self.regularisation_emissions = regularisation_emissions
        self.regularisation_transitions = regularisation_transitions
        self.regularisation_transitions_order_2 = regularisation_transitions_order_2

    def _produce_embeddings(self, sentences, mask):

        # Produce context-aware embeddings, with positional embeddings
        if self._has_country_classifier:
            embeddings = self.embedding(sentences) + self.pos_embed[:, 1:, :]

            # Add positional embeddings to CLS tokens
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            # Clone the CLS token to match the number of embeddings
            cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)

            # Prepend CLS tokens to the embeddings sequence
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

            # Take cls token into account in the mask
            true_mask = torch.ones(embeddings.shape[0], 1, dtype=torch.bool, device=embeddings.device)
            mask = torch.cat((true_mask, mask), dim=1)
        else:
            embeddings = self.embedding(sentences) + self.pos_embed

        # Self-attention layers
        embeddings = self.transformer_encoder(
            src=embeddings, src_key_padding_mask=~mask
        )

        return embeddings

    def _produce_emissions(self, embeddings):
        """Produce emissions (i.e., logits) for each token in each sentence."""

        # Compute emissions
        emissions = self.projection(embeddings)

        return emissions

    def loss(self, sentences, gt_country_ids, tags, mask):
        """Compute the negative log-likelihood of the output of the model with respect to the ground truth tags."""

        # Compute embeddings
        embeddings = self._produce_embeddings(sentences, mask)

        # Extract CLS token if present
        if self._has_country_classifier:
            cls_token, embeddings = embeddings[:, 0, :], embeddings[:, 1:, :]

        # (1) Country loss: Cross entropy on the predicted country distributions
        country_loss = 0
        if self._has_country_classifier:
            country_loss = self.country_predictor.loss(cls_token, gt_country_ids)

        # (2) Structural loss: negative log likelihood from the crf predictions
        emissions = self._produce_emissions(embeddings)
        negative_log_likelihood = -self.crf(emissions, tags, mask)

        # (3) Regularisation
        reg = self.regularisation_emissions * torch.sum(emissions ** 2)
        reg += self.regularisation_transitions * torch.sum(self.crf.transitions ** 2)
        with suppress(RuntimeError):
            reg += self.regularisation_transitions_order_2 * torch.sum(self.crf.transitions_order_2 ** 2)

        return country_loss, negative_log_likelihood, reg

    def decode(self, sentences, mask, emissions_forced=None):
        """Decode (i.e., predict) the most likely tags for each token in each sentence.
        Also returns the emissions.
        Can force the emissions and bypass the transformer."""

        embeddings = self._produce_embeddings(sentences, mask)

        cls_token = None
        if self._has_country_classifier:
            cls_token, embeddings = embeddings[:, 0, :], embeddings[:, 1:, :]  # Remove CLS token

        emissions = self._produce_emissions(embeddings)

        if emissions_forced is not None:
            emissions = emissions_forced

        tags_estimated = self.crf.decode(emissions, mask=mask)

        return tags_estimated, emissions, cls_token, embeddings

    def forward(self, sentences, gt_country_ids=None, tags=None, mask=None):
        """Perform inference if tags=None, or training (i.e., ret|Gurn the loss) if tags=True."""
        if tags is None:  # Inference (no ground truth tags are given)
            tags_estimated, _, cls_token, _ = self.decode(sentences, mask)
            country_preds_indices, country_preds_confidences = None, None
            if self._has_country_classifier:
                country_preds_indices, country_preds_confidences = self.country_predictor.predict(cls_token)
            return tags_estimated, country_preds_indices, country_preds_confidences
        # Training (ground truth tags are given)
        return self.loss(sentences, gt_country_ids, tags, mask)

    def predict_tags(self,
                     sentences: list[str],
                     tokenizer,
                     strict_before_inside: bool = True,
                     emissions_forced=None
                     ) -> list[ResultCRF]:

        self.eval()
        all_results = []
        with torch.inference_mode():
            # Encode to ids
            all_word_ids = [tokenizer.encode(sentence) for sentence in sentences]
            # Padding
            all_padded_words_ids = [(word_ids + (self._max_seq_len - len(word_ids)) * [self._padding_idx])
                                    for word_ids in all_word_ids]

            padded_words_ids = torch.tensor(all_padded_words_ids, device=self.pos_embed.device)

            # Create Mask
            mask = (padded_words_ids != self._padding_idx).to(self.pos_embed.device)

            # Produce tags
            tags_estimated, emissions, cls_token, _ = self.decode(padded_words_ids,
                                                                  mask,
                                                                  emissions_forced=emissions_forced)

            all_tags = [[self._idx_to_tag[idx] for idx in tags_ids] for tags_ids in tags_estimated]

            # Produce GT country
            country_preds_indices, country_preds_confidences = None, None
            if self._has_country_classifier:
                country_preds_indices, country_preds_confidences \
                    = self.country_predictor.predict(sentence_embedding=cls_token)
                country_preds_indices = country_preds_indices.tolist()
                country_preds_confidences = country_preds_confidences.tolist()

            # If we use a CRF class that supports it, return the marginal probability matrix
            log_likelihood = self.crf.marginal_probabilities(emissions)

            # Post-process BIO tags to output tags.
            for i, _ in enumerate(all_tags):
                if len(all_tags[i]):
                    grouped_tags = create_details_from_biotags(
                        raw_content=sentences[i],
                        country=self._mapping_id_to_country[
                            country_preds_indices[i]
                        ] if country_preds_indices is not None else None,
                        country_confidence=country_preds_confidences[i] if country_preds_confidences else None,
                        tags=all_tags[i],
                        strict_before_inside=strict_before_inside
                    )
                    all_results.append(
                        ResultCRF(grouped_tags=grouped_tags,
                                  emissions=emissions[i, :, :],
                                  log_likelihood=log_likelihood[:, i, :]
                                  )
                    )

            return all_results
