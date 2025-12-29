"""
Module providing the CRF runner.
"""
import orjson
import logging
from collections import defaultdict
from typing import Iterator

import safetensors.torch
import torch

from data_structuring.components.models import TransformerCRF
from data_structuring.components.runners.base_runner import BaseRunner
from data_structuring.components.runners.result_processing import ResultRunnerCRF, PredictionCRF
from data_structuring.components.tags import Tag, BIOTag
from data_structuring.components.tokenizers import CharacterTokenizer

from data_structuring.components.database import Database
from data_structuring.config import CRFConfig


logger = logging.getLogger(__name__)


class RunnerCRF(BaseRunner):
    """
    This class is a wrapper around the TransformerCRF module to make predictions with the CRF x Transformer model.
    """

    def __init__(self, config: CRFConfig, database: Database):
        super().__init__(config=config, database=database)
        # Fetch model weights
        model_weights = safetensors.torch.load_file(self.config.model_weights_path)
        # Check if checkpoint contains country predictor
        use_country_predictor = any("country_predictor" in weight_name for weight_name in model_weights)
        # Load model configuration from JSON file
        with open(self.config.model_config_path, "r") as fp:
            self.model_config = orjson.loads(fp.read())
        # Convert tags to correct type
        self.model_config['mapping_id_to_country'] = {int(key): iso for key, iso in
                                                      self.model_config['mapping_id_to_country'].items()}
        self.model_config['tags_to_keep'] = [Tag(tag) for tag in self.model_config['tags_to_keep']]
        self.model_config['bio_tags_to_keep'] = [BIOTag(**tag) for tag in self.model_config['bio_tags_to_keep']]

        self.tokenizer = CharacterTokenizer(self.model_config["vocabulary"])
        self.model = TransformerCRF(
            vocab_size=self.tokenizer.vocab_size,
            tags=self.model_config["bio_tags_to_keep"],
            mapping_id_to_country=self.model_config["mapping_id_to_country"],
            max_seq_len=self.model_config["max_sequence_length"],
            d_model=self.model_config["embedding_dimension"],
            nhead=self.model_config["n_heads"],
            depth=self.model_config["depth"],
            padding_idx=self.model_config["padding_value"],
            use_country_classifier=use_country_predictor,
            regularisation_emissions=self.model_config["regularisation_emissions"],
            regularisation_transitions=self.model_config["regularisation_transitions"],
            regularisation_transitions_order_2=self.model_config["regularisation_transitions_order_2"]
        ).to(self.config.device)
        self.model.load_state_dict(model_weights)

    def _get_log_probas_and_emissions_B_I(self, tag, emissions, marginal_probas):

        # Tag.OTHER cannot be BIO, treat it separately
        if tag == Tag.OTHER:
            row_number = self.model_config["bio_tags_to_keep"].index(BIOTag.create_other())
            saved_emissions = emissions[:, row_number]
            saved_marginal_probas = marginal_probas[:, row_number]
        else:
            row_numbers = (
                self.model_config["bio_tags_to_keep"].index(BIOTag.create_before(tag)),
                self.model_config["bio_tags_to_keep"].index(BIOTag.create_inside(tag)),
            )
            saved_emissions = emissions[:, row_numbers[0]] + emissions[:, row_numbers[1]]
            saved_marginal_probas = marginal_probas[:, row_numbers[0]] + marginal_probas[:, row_numbers[1]]

        return saved_marginal_probas, saved_emissions

    def tag(self, data: list[str]) -> Iterator[ResultRunnerCRF]:

        logger.info("Start CRF tagging")
        predictions = self.model.predict_tags(data, tokenizer=self.tokenizer, strict_before_inside=False)

        # Post-process raw CRF results into RunnerCRF results
        for result, sample in zip(predictions, data):

            # Group predictions/log probas/emissions of the same tags
            predictions_per_tag = defaultdict(set)
            logprobs_per_tag = {}
            emissions_per_tag = {}

            # Populate
            for span in result.grouped_tags.spans:
                logprobs_per_tag[span.tag], emissions_per_tag[span.tag] = (
                    self._get_log_probas_and_emissions_B_I(span.tag, result.emissions, result.log_likelihood))
                predictions_per_tag[span.tag].add(
                    PredictionCRF(
                        start=span.start,
                        end=span.end,
                        tag=span.tag,
                        prediction=sample[span.start:span.end],
                        confidence=logprobs_per_tag[span.tag][span.start:span.end].mean()
                    )
                )

            # Set default values
            for tag in self.model_config["tags_to_keep"]:
                if tag not in predictions_per_tag:
                    predictions_per_tag[tag] = set()
                    logprobs_per_tag[tag] = torch.zeros(self.model_config["max_sequence_length"],
                                                        device=result.emissions.device)
                    emissions_per_tag[tag] = torch.zeros(self.model_config["max_sequence_length"],
                                                         device=result.emissions.device)

            yield ResultRunnerCRF(
                details=result.grouped_tags,
                predictions_per_tag=predictions_per_tag,
                emissions_per_tag=emissions_per_tag,
                log_probas_per_tag=logprobs_per_tag
            )
        logger.info("Done CRF tagging")
