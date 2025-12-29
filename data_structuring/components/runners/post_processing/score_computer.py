"""Score computation - Single Responsibility: Calculate final scores for matches."""

import math
import numpy as np

from data_structuring.components.flags import TownFlag, CountryFlag, CommonFlag, BaseFlag
from data_structuring.config import PostProcessingTownWeightsConfig, PostProcessingCountryWeightsConfig


class ScoreComputer:
    """Computes final scores for town and country matches."""

    def __init__(self,
                 town_weights: PostProcessingTownWeightsConfig,
                 country_weights: PostProcessingCountryWeightsConfig):
        self.town_weights = town_weights
        self.country_weights = country_weights

    def compute_town_score(self, crf_score: float, dist_score: int, flags: list[BaseFlag]) -> float:
        """
        Compute final score for a town match.

        Args:
            crf_score: CRF probability score in [0, 1]
            dist_score: Fuzzy match distance
            flags: List of flags for this match

        Returns:
            Final score in [0, 1]
        """
        eps: float = 1e-6
        crf_max_contribution: float = 0.9
        min_bonus_mul: float = 2.5
        max_bonus_mul: float = 4.0
        # Clip and convert to log-odds
        crf_score = np.clip(crf_score, a_min=eps, a_max=1 - eps)
        amortized_crf_score = np.clip(crf_score, a_min=1 - crf_max_contribution, a_max=crf_max_contribution)
        amortized_log_odds = math.log(amortized_crf_score / (1. - amortized_crf_score))

        # Calculate bonuses
        bonuses_sum = sum([
            self.town_weights.is_in_last_third if CommonFlag.IS_IN_LAST_THIRD in flags else 0,
            self.town_weights.could_be_reasonable_mistake if CommonFlag.COULD_BE_REASONABLE_MISTAKE in flags else 0,
            self.town_weights.country_is_present_bonus if TownFlag.COUNTRY_IS_PRESENT in flags else 0,
            self.town_weights.mlp_country_is_present_bonus if TownFlag.MLP_COUNTRY_IS_PRESENT in flags else 0,
            self.town_weights.is_very_close_to_country if TownFlag.IS_VERY_CLOSE_TO_COUNTRY in flags else 0,
            self.town_weights.is_on_same_line_as_country if TownFlag.IS_ON_SAME_LINE_AS_COUNTRY in flags else 0,
            self.town_weights.postcode_for_town_found if TownFlag.POSTCODE_FOR_TOWN_FOUND in flags else 0,
            self.town_weights.is_metropolis if TownFlag.IS_METROPOLIS in flags else 0,
            self.town_weights.is_alone_on_line if TownFlag.IS_ALONE_ON_LINE in flags else 0
        ])

        # Calculate maluses
        maluses_sum = sum([
            self.town_weights.contains_typo * dist_score if (
                CommonFlag.IS_SEPARATOR_TYPO not in flags) else 0,
            self.town_weights.is_inside_another_word if (
                CommonFlag.IS_INSIDE_ANOTHER_WORD in flags) else 0,
            self.town_weights.is_in_first_third if (
                CommonFlag.IS_IN_FIRST_THIRD in flags) else 0,
            self.town_weights.is_short if (
                CommonFlag.IS_SHORT in flags) else 0,
            self.town_weights.is_inside_another_lower_ranked_match if (
                CommonFlag.IS_INSIDE_ANOTHER_LOWER_RANKED_MATCH in flags) else 0,
            self.town_weights.is_small_town if (
                TownFlag.IS_SMALL_TOWN in flags) else 0,
            self.town_weights.is_small_town_and_country_not_present if (
                TownFlag.IS_SMALL_TOWN in flags and TownFlag.COUNTRY_IS_PRESENT not in flags) else 0,
            self.town_weights.country_is_present_malus if (
                TownFlag.COUNTRY_IS_PRESENT not in flags) else 0,
            self.town_weights.is_from_extended_data if (
                TownFlag.IS_FROM_EXTENDED_DATA in flags) else 0,
            self.town_weights.is_not_largest_town_with_name if (
                TownFlag.IS_NOT_LARGEST_TOWN_WITH_NAME in flags) else 0,
            self.town_weights.is_inside_street if (
                CommonFlag.IS_INSIDE_STREET in flags) else 0,
            self.town_weights.is_common_state_province_alias if (
                CommonFlag.IS_COMMON_STATE_PROVINCE_ALIAS in flags) else 0,
            self.town_weights.is_uncommon_state_province_alias if (
                CommonFlag.IS_UNCOMMON_STATE_PROVINCE_ALIAS in flags) else 0,
            self.town_weights.is_short_and_nonzero_dist_score if (
                CommonFlag.IS_SHORT in flags and dist_score > 0) else 0,
            self.town_weights.is_short_and_is_inside_another_word if (
                CommonFlag.IS_SHORT in flags and CommonFlag.IS_INSIDE_ANOTHER_WORD in flags) else 0,
            self.town_weights.is_inside_another_higher_ranked_match if (
                CommonFlag.IS_INSIDE_ANOTHER_HIGHER_RANKED_MATCH in flags) else 0
        ])

        # Apply bonuses/maluses in log space
        bonus_mul = max_bonus_mul - (max_bonus_mul - min_bonus_mul) * crf_score
        malus_mul = min_bonus_mul + (max_bonus_mul - min_bonus_mul) * crf_score
        log_odds = amortized_log_odds + bonus_mul * bonuses_sum + malus_mul * maluses_sum

        # Convert back to probability
        return 1. / (1. + math.exp(-log_odds))

    def compute_country_score(self, crf_score: float, dist_score: int, flags: list[BaseFlag]) -> float:
        """
        Compute final score for a country match.

        Args:
            crf_score: CRF probability score in [0, 1]
            dist_score: Fuzzy match distance
            flags: List of flags for this match

        Returns:
            Final score in [0, 1]
        """
        eps: float = 1e-6
        crf_max_contribution: float = 0.9
        min_bonus_mul: float = 2.5
        max_bonus_mul: float = 4.0

        # Clip and convert to log-odds
        crf_score = np.clip(crf_score, a_min=eps, a_max=1 - eps)
        amortized_crf_score = np.clip(crf_score, a_min=1 - crf_max_contribution, a_max=crf_max_contribution)
        amortized_log_odds = math.log(amortized_crf_score / (1. - amortized_crf_score))

        # Calculate bonuses
        bonuses_sum = sum([
            self.country_weights.is_in_last_third if CommonFlag.IS_IN_LAST_THIRD in flags else 0,
            self.country_weights.could_be_reasonable_mistake if CommonFlag.COULD_BE_REASONABLE_MISTAKE in flags else 0,
            self.country_weights.town_is_present if CountryFlag.TOWN_IS_PRESENT in flags else 0,
            self.country_weights.is_very_close_to_town if CountryFlag.IS_VERY_CLOSE_TO_TOWN in flags else 0,
            self.country_weights.is_on_same_line_as_town if CountryFlag.IS_ON_SAME_LINE_AS_TOWN in flags else 0,
            self.country_weights.postal_code_is_present if CountryFlag.POSTAL_CODE_IS_PRESENT in flags else 0,
            self.country_weights.iban_is_present if CountryFlag.IBAN_IS_PRESENT in flags else 0,
            self.country_weights.phone_prefix_is_present if CountryFlag.PHONE_PREFIX_IS_PRESENT in flags else 0,
            self.country_weights.domain_is_present if CountryFlag.DOMAIN_IS_PRESENT in flags else 0,
            self.country_weights.mlp_strongly_agrees if CountryFlag.MLP_STRONGLY_AGREES in flags else 0,
            self.country_weights.mlp_agrees if CountryFlag.MLP_AGREES in flags else 0,
            self.country_weights.mlp_doesnt_disagree if CountryFlag.MLP_DOESNT_DISAGREE in flags else 0
        ])

        # Calculate maluses
        maluses_sum = sum([
            self.country_weights.contains_typo * dist_score if (
                CommonFlag.IS_SEPARATOR_TYPO not in flags) else 0,
            self.country_weights.is_inside_another_word if (
                CommonFlag.IS_INSIDE_ANOTHER_WORD in flags) else 0,
            self.country_weights.is_in_first_third if (
                CommonFlag.IS_IN_FIRST_THIRD in flags) else 0,
            self.country_weights.is_short if CommonFlag.IS_SHORT in flags else 0,
            self.country_weights.is_inside_another_lower_ranked_match if (
                CommonFlag.IS_INSIDE_ANOTHER_LOWER_RANKED_MATCH in flags) else 0,
            self.country_weights.is_inside_street if (
                CommonFlag.IS_INSIDE_STREET in flags) else 0,
            self.country_weights.is_common_state_province_alias if (
                CommonFlag.IS_COMMON_STATE_PROVINCE_ALIAS in flags) else 0,
            self.country_weights.is_uncommon_state_province_alias if (
                CommonFlag.IS_UNCOMMON_STATE_PROVINCE_ALIAS in flags) else 0,
            self.country_weights.is_short_and_nonzero_dist_score if (
                CommonFlag.IS_SHORT in flags and dist_score > 0) else 0,
            self.country_weights.is_short_and_is_inside_another_word if (
                CommonFlag.IS_SHORT in flags and CommonFlag.IS_INSIDE_ANOTHER_WORD in flags) else 0,
            self.country_weights.is_inside_another_higher_ranked_match if (
                CommonFlag.IS_INSIDE_ANOTHER_HIGHER_RANKED_MATCH in flags) else 0
        ])

        # Apply bonuses/maluses in log space
        bonus_mul = max_bonus_mul - (max_bonus_mul - min_bonus_mul) * crf_score
        malus_mul = min_bonus_mul + (max_bonus_mul - min_bonus_mul) * crf_score
        log_odds = amortized_log_odds + bonus_mul * bonuses_sum + malus_mul * maluses_sum

        # Convert back to probability
        return 1. / (1. + math.exp(-log_odds))
