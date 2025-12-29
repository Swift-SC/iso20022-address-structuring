"""
Module providing the FuzzyMatching runner.
"""
import logging
from typing import Generator

from data_structuring.components.fuzzy_matching import fuzzy_scan
from data_structuring.components.database import Database
from data_structuring.components.runners.base_runner import BaseRunner
from data_structuring.components.runners.result_processing import ResultRunnerFuzzyMatch
from data_structuring.config import FuzzyMatchConfig


logger = logging.getLogger(__name__)


class RunnerFuzzyMatch(BaseRunner):
    """
    This class is a wrapper around the FuzzyMatching module.
    """

    def __init__(self, config: FuzzyMatchConfig, database: Database):
        super().__init__(config=config, database=database)

    def match(self, data: list[str]) -> Generator[ResultRunnerFuzzyMatch, None, None]:
        logger.info("Start batched fuzzy matching runner")
        # Start batched fuzzy matching on countries, country codes, and towns
        logger.info("Countries batch")
        all_fs_countries = fuzzy_scan.fuzzyscan_all_batched(data,
                                                            self.database.all_possibilities_country,
                                                            score_cutoff=self.config.fuzzy_match_score_cutoff,
                                                            max_l_dist=self.config.fuzzy_match_tolerance,
                                                            n_workers=self.config.num_workers)

        logger.info("Country codes batch")
        all_fs_country_codes = fuzzy_scan.fuzzyscan_all_batched(data,
                                                                self.database.alpha2_to_alpha2,
                                                                max_l_dist=0,  # Exact matching for country codes
                                                                score_cutoff=100,  # Exact matching for country codes
                                                                n_workers=self.config.num_workers)

        logger.info("Towns batch")
        all_fs_towns = fuzzy_scan.fuzzyscan_all_batched(data,
                                                        self.database.all_possibilities_town,
                                                        score_cutoff=self.config.fuzzy_match_score_cutoff_towns,
                                                        max_l_dist=self.config.fuzzy_match_tolerance_towns,
                                                        n_workers=self.config.num_workers)

        # Same principle for extended towns
        all_word_mapping_towns_extended = [
            {
                key: value for iso in
                set([match.origin for match in fuzzy_scan.FuzzyMatchResult.merge(fs_countries, fs_country_code)
                     if match.origin is not None] + self.database.countries_overrides)
                if iso in self.database.extended_all_possibilities_town
                for key, value in self.database.extended_all_possibilities_town[iso].items()
            }
            for fs_countries, fs_country_code in zip(all_fs_countries, all_fs_country_codes)
        ]

        all_extended_fs_towns = [
            fuzzy_scan.fuzzyscan_all_batched([sample],
                                             word_mapping_towns_extended,
                                             score_cutoff=self.config.fuzzy_match_score_cutoff_towns,
                                             max_l_dist=self.config.fuzzy_match_tolerance_towns,
                                             n_workers=self.config.num_workers)
            for sample, word_mapping_towns_extended in zip(data, all_word_mapping_towns_extended)
        ]

        # Group results by sample
        for country_res, country_code_res, town_res, extended_town_res in (zip(all_fs_countries,
                                                                               all_fs_country_codes, all_fs_towns,
                                                                               all_extended_fs_towns)):

            yield ResultRunnerFuzzyMatch(
                country_matches=country_res,
                country_code_matches=country_code_res,
                town_matches=town_res,
                extended_town_matches=extended_town_res[0])
        logger.info("Done fuzzy matching")
