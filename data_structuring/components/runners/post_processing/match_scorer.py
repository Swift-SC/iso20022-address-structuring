"""
Match scoring operations - Single Responsibility: Score matches with CRF emissions.
"""

import numpy as np
from data_structuring.components.fuzzy_matching.fuzzy_scan import FuzzyMatchResult


class MatchScorer:
    """Scores fuzzy matches using CRF emission probabilities."""

    @staticmethod
    def score_matches_with_emissions(fuzzy_match_result: FuzzyMatchResult,
                                     marginal_logprobability_by_token: np.ndarray,
                                     emission_by_token: np.ndarray
                                     ) -> None:
        """
        Score each match based on average CRF scores for its token positions.

        Args:
            fuzzy_match_result: List of fuzzy matches to score
            marginal_logprobability_by_token: Log probabilities per token
            emission_by_token: Raw emission scores per token
        """
        for match in fuzzy_match_result:
            match.crf_score = marginal_logprobability_by_token[match.start: match.end].mean()
            match.transformer_score = emission_by_token[match.start: match.end].mean()
