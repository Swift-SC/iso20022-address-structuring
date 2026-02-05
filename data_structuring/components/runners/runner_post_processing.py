"""
Module providing the post-processing runner.
"""
from __future__ import annotations

import logging
import re
from typing import Generator

import numpy as np

from data_structuring.components.data_provider import generate_duplicate_aliases
from data_structuring.components.database import Database
from data_structuring.components.flags import TownFlag, CommonFlag, CountryFlag
from data_structuring.components.fuzzy_matching.fuzzy_scan import FuzzyMatch, FuzzyMatchResult
from data_structuring.components.readers.base_reader import AddressSample
from data_structuring.components.runners.base_runner import BaseRunner
from data_structuring.components.runners.post_processing.combination_generator import CombinationGenerator
from data_structuring.components.runners.post_processing.flag_managers import (
    TownFlagManager,
    CountryFlagManager,
    RelationshipFlagManager,
    MatchInclusionFlagger
)
from data_structuring.components.runners.post_processing.match_scorer import MatchScorer
from data_structuring.components.runners.post_processing.score_computer import ScoreComputer
from data_structuring.components.runners.result_processing import (
    ResultRunnerCRF,
    ResultRunnerFuzzyMatch,
    ResultPostProcessing
)
from data_structuring.components.runners.runner_postcode_match import ResultRunnerPostcodeMatch
from data_structuring.components.tags import Tag
from data_structuring.config import (
    PostProcessingConfig,
    PostProcessingTownWeightsConfig,
    PostProcessingCountryWeightsConfig
)

logger = logging.getLogger(__name__)


class RunnerPostProcessing(BaseRunner):
    """
    Orchestrates post-processing of CRF and fuzzy match results.
    """

    def __init__(self,
                 config: PostProcessingConfig,
                 town_weights: PostProcessingTownWeightsConfig,
                 country_weights: PostProcessingCountryWeightsConfig,
                 database: Database):
        """Initialize orchestrator with configuration and dependencies."""
        super().__init__(config=config, database=database)
        self.town_weights = town_weights
        self.country_weights = country_weights

        # Initialize SOLID components
        self.match_scorer = MatchScorer()
        self.town_flag_manager = TownFlagManager(database, config)
        self.country_flag_manager = CountryFlagManager(database, config)
        self.relationship_manager = RelationshipFlagManager(database)
        self.inclusion_flagger = MatchInclusionFlagger()
        self.score_computer = ScoreComputer(town_weights, country_weights)
        self.combination_generator = CombinationGenerator(database, config, town_weights, country_weights)

    def filter_fuzzy_results(self,
                             fuzzymatch_result: FuzzyMatchResult,
                             original_text_length: int
                             ) -> None:
        """
        Filter and add flags to fuzzymatch_result, sorting it in the process.

        Args:
            fuzzymatch_result: list to be filtered and sorted. It will be modified in place.
            original_text_length: length of the original text

        Returns:
            None
        """
        # Sort by distance and CRF score
        fuzzymatch_result.sort(key=lambda x: (-x.dist, x.crf_score), reverse=True)

        # Flag inclusions
        self.inclusion_flagger.flag_matches_included_in_another(fuzzymatch_result, fuzzymatch_result)

        one_third = original_text_length / 3
        two_third = original_text_length * 2 / 3
        # Add position-based flags
        for match in fuzzymatch_result:
            is_short = match.end - match.start <= 2
            is_in_last_third = match.start >= two_third
            is_in_first_third = match.start <= one_third

            if is_short:
                match.flags.append(CommonFlag.IS_SHORT)
            if is_in_first_third:
                match.flags.append(CommonFlag.IS_IN_FIRST_THIRD)
            if is_in_last_third:
                match.flags.append(CommonFlag.IS_IN_LAST_THIRD)

    def split_country_list_in_code_and_not_code(self,
                                                country_list: FuzzyMatchResult
                                                ) -> tuple[list[FuzzyMatch], list[FuzzyMatch]]:
        """Split country list into codes (<=2 chars) and full names (>2 chars)."""
        code = []
        not_code = []

        for country in country_list:
            if len(country.possibility) <= 2:
                code.append(country)
            else:
                not_code.append(country)

        return code, not_code

    def run(self,
            crf_results: list[ResultRunnerCRF],
            fuzzy_match_results: list[ResultRunnerFuzzyMatch],
            postcode_match_results: list[ResultRunnerPostcodeMatch],
            address_samples: list[AddressSample]
            ) -> Generator[ResultPostProcessing, None, None]:
        """Post-process CRF and fuzzy match results to generate final predictions."""

        logger.info("Start post-processing runner")
        for crf_result, fuzzy_match_result, postcode_match_result, address_sample in zip(
                crf_results, fuzzy_match_results, postcode_match_results, address_samples):

            sample = crf_result.details.content
            sample_casefold = sample.casefold()
            sample_length = len(sample)

            suggested_country = address_sample.suggested_country
            force_suggested_country = address_sample.force_suggested_country

            # Set suggested_country and force_suggested_country in the results combination generator
            self.combination_generator.set_suggested_country(suggested_country)
            self.combination_generator.set_force_suggested_country(force_suggested_country)

            # Extract marginal probabilities from CRF
            country_marginal_emission_by_token = (crf_result.emissions_per_tag[Tag.COUNTRY]
                                                  .cpu()
                                                  .numpy()
                                                  .astype(np.float64))
            country_marginal_logprobability_by_token = (crf_result.log_probas_per_tag[Tag.COUNTRY]
                                                        .cpu()
                                                        .numpy()
                                                        .astype(np.float64))
            town_marginal_emission_by_token = (crf_result.emissions_per_tag[Tag.TOWN]
                                               .cpu()
                                               .numpy()
                                               .astype(np.float64))
            town_marginal_logprobability_by_token = (crf_result.log_probas_per_tag[Tag.TOWN]
                                                     .cpu()
                                                     .numpy()
                                                     .astype(np.float64))

            # Score country matches with CRF emissions using MatchScorer
            self.match_scorer.score_matches_with_emissions(
                fuzzy_match_result.country_matches,
                country_marginal_logprobability_by_token,
                country_marginal_emission_by_token
            )
            self.match_scorer.score_matches_with_emissions(
                fuzzy_match_result.country_code_matches,
                country_marginal_logprobability_by_token,
                country_marginal_emission_by_token
            )

            if suggested_country:
                best_crf_score = self.config.base_score_suggested_country
                for match in fuzzy_match_result.country_code_matches:
                    if suggested_country == match.possibility and suggested_country == match.origin:
                        match.flags.append(CountryFlag.IS_SUGGESTED_COUNTRY)
                        match.crf_score = max(match.crf_score, self.config.base_score_suggested_country)
                        best_crf_score = max(match.crf_score, best_crf_score)

                suggested_country_match = FuzzyMatch(start=len(address_sample.text) + 2,
                                                     end=len(address_sample.text) + 2,
                                                     matched="", dist=0, origin=suggested_country,
                                                     possibility=suggested_country,
                                                     crf_score=best_crf_score,
                                                     flags=[CountryFlag.GENERATED_BY_SUGGESTED_COUNTRY])
                fuzzy_match_result.country_code_matches = FuzzyMatchResult.merge(
                    fuzzy_match_result.country_code_matches, FuzzyMatchResult([suggested_country_match]))

            # Filter country codes with zero score and merge
            fuzzy_match_result.country_code_matches = FuzzyMatchResult([
                match for match in fuzzy_match_result.country_code_matches
                if match.crf_score > 0 or match.origin == crf_result.details.country_code
            ])
            fuzzy_match_result.country_matches = FuzzyMatchResult.merge(
                fuzzy_match_result.country_matches,
                fuzzy_match_result.country_code_matches
            )

            # Process town matches
            for match in fuzzy_match_result.extended_town_matches:
                match.flags.append(TownFlag.IS_FROM_EXTENDED_DATA)

            fuzzy_match_result.town_matches = FuzzyMatchResult.merge(fuzzy_match_result.town_matches,
                                                                     fuzzy_match_result.extended_town_matches)
            self.match_scorer.score_matches_with_emissions(fuzzy_match_result.town_matches,
                                                           town_marginal_logprobability_by_token,
                                                           town_marginal_emission_by_token)

            # Add flags using FlagManagers
            ibans = re.findall(self.config.iban_pattern, sample)
            self.town_flag_manager.add_all_flags(fuzzy_match_result, crf_result)
            self.country_flag_manager.add_all_flags(fuzzy_match_result, crf_result, sample, sample_casefold, ibans)

            # Filter fuzzy results
            self.filter_fuzzy_results(fuzzy_match_result.country_matches, sample_length)
            self.filter_fuzzy_results(fuzzy_match_result.town_matches, sample_length)

            # Remove country codes within full country matches using MatchInclusionFlagger
            country_codes_result_list, non_country_codes_result_list = (
                self.split_country_list_in_code_and_not_code(fuzzy_match_result.country_matches))
            self.inclusion_flagger.flag_matches_included_in_another(
                queries=country_codes_result_list,
                larger_matches=fuzzy_match_result.country_matches)
            fuzzy_match_result.country_matches = FuzzyMatchResult(
                country_codes_result_list + non_country_codes_result_list)

            # Add country-town relationship flags using RelationshipFlagManager
            country_head = (crf_result.details.country_code
                            if (crf_result.details.country_code
                                and crf_result.details.country_code_confidence * 100.0 >= 0.99)
                            else None)
            self.relationship_manager.add_relationship_flags(fuzzy_match_result, sample, country_head)
            self.relationship_manager.check_reasonable_mistakes(fuzzy_match_result, crf_result)
            self.town_flag_manager.check_alone_on_line(fuzzy_match_result, sample)

            # Check postcode matches
            for postal_result in postcode_match_result.postcode_matches:
                for town_match in fuzzy_match_result.town_matches:
                    for postal_result_town_alias in generate_duplicate_aliases(postal_result.possibility):
                        if (postal_result_town_alias == town_match.possibility
                                and postal_result.origin == town_match.origin):
                            town_match.flags.append(TownFlag.POSTCODE_FOR_TOWN_FOUND)

            # Compute final scores using ScoreComputer
            for country_result in fuzzy_match_result.country_matches:
                # Remove all flags for the synthetically generated suggested country match
                if CountryFlag.GENERATED_BY_SUGGESTED_COUNTRY in country_result.flags:
                    country_result.flags = [CountryFlag.GENERATED_BY_SUGGESTED_COUNTRY]
                # Deduplicate and sort flags
                country_result.flags = list(set(country_result.flags))
                country_result.flags.sort()
                country_result.final_score = self.score_computer.compute_country_score(
                    country_result.crf_score,
                    country_result.dist,
                    country_result.flags
                )
            for town_result in fuzzy_match_result.town_matches:
                # Deduplicate and sort flags
                town_result.flags = list(set(town_result.flags))
                town_result.flags.sort()
                town_result.final_score = self.score_computer.compute_town_score(
                    town_result.crf_score,
                    town_result.dist,
                    town_result.flags
                )

            # Generate country-town combinations using CombinationGenerator
            no_country = FuzzyMatch(start=0, end=0, matched="", dist=0, origin="NO COUNTRY",
                                    possibility="NO COUNTRY",
                                    final_score=self.config.minimal_final_score_country)
            no_town = FuzzyMatch(start=0, end=0, matched="", dist=0, origin="",
                                 possibility="NO TOWN",
                                 final_score=self.config.minimal_final_score_town)

            countries_above_threshold = [
                match for match in fuzzy_match_result.country_matches
                if match.final_score >= self.config.minimal_final_score_country
            ]

            towns_above_threshold = [
                match for match in fuzzy_match_result.town_matches
                if match.final_score >= self.config.minimal_final_score_town
            ]

            country_town_combinations = self.combination_generator.generate_combinations(
                countries_above_threshold,
                towns_above_threshold,
                no_country,
                no_town
            )

            # Order results by combined scores
            fuzzy_match_result.country_matches = FuzzyMatchResult(
                [match[0] for match in country_town_combinations])
            fuzzy_match_result.town_matches = FuzzyMatchResult(
                [match[1] for match in country_town_combinations])

            yield ResultPostProcessing(
                crf_result=crf_result,
                fuzzy_match_result=fuzzy_match_result,
                ibans=ibans,
                suggested_country=suggested_country,
                force_suggested_country=force_suggested_country
            )
        logger.info("Done post-processing")
