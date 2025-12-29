"""
This module contains the CountryHead class for country prediction in address parsing.

The CountryHead is a neural network module that implements a multi-layer perceptron (MLP)
to predict the country of origin for a given sentence embedding. It takes sentence-level
embeddings as input and outputs country predictions with associated confidence scores.

The module supports both training (with loss computation) and inference (with probability
predictions and top-k results).
"""
import torch
from torch import nn, topk


class CountryHead(nn.Module):
    """
    A simple country prediction head implemented as an MLP.
    """

    def __init__(self, embedding_dim: int, num_countries: int, mlp_factor: float = 2.0):
        super().__init__()
        self.hidden_dim = int(embedding_dim * mlp_factor)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, self.hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(self.hidden_dim, num_countries))
        self._loss = nn.CrossEntropyLoss()

    def forward(self, sentence_embedding):
        return self.mlp(sentence_embedding)

    def loss(self, sentence_embedding, gt):
        preds = self(sentence_embedding)
        return self._loss(preds, gt)

    @torch.no_grad()
    def predict_probs(self, sentence_embedding, k=3):
        logits = self(sentence_embedding)
        normalized_logits = nn.functional.softmax(logits, dim=1)
        top_k = topk(normalized_logits, k=k, dim=1)
        return top_k.indices, top_k.values

    @torch.no_grad()
    def predict(self, sentence_embedding):
        indices, values = self.predict_probs(sentence_embedding, k=1)
        return indices.squeeze(1), values.squeeze(1)
