"""
Module providing the postcode matching runner.
"""
from functools import reduce, partial
from typing import Generator
import logging

from data_structuring.components.database import Database
from data_structuring.components.post_code_matching import post_code_match
from data_structuring.components.runners.base_runner import BaseRunner
from data_structuring.components.runners.result_processing import ResultRunnerPostcodeMatch
from data_structuring.config import FuzzyMatchConfig

AR_POSTCODE_STRUCTURE = "[0-9]{4}"
BR_POSTCODE_STRUCTURE = "[0-9]{3}"
CL_POSTCODE_STRUCTURE = "[0-9]{4}"
CN_POSTCODE_STRUCTURE = "[0-9]{2}"
IE_POSTCODE_STRUCTURE = "(?:[a-zA-Z][0-9][a-zA-Z][0-9]|[a-zA-Z]{2}[0-9]{2}|[a-zA-Z][0-9][a-zA-Z]{2})"
MT_POSTCODE_STRUCTURE = "[0-9]{4}"


logger = logging.getLogger(__name__)


class RunnerPostcodeMatch(BaseRunner):
    """
    This class is a wrapper around the post_code_matching module.
    """

    def __init__(self, config: FuzzyMatchConfig, database: Database):
        super().__init__(config=config, database=database)

    def match(self, data: list[str]) -> Generator[ResultRunnerPostcodeMatch, None, None]:
        logger.info("Start Postcode matching runner")
        almost_all_postcodes_match = partial(post_code_match.find_postcode_town_matches,
                                             postcodes_dict=self.database.full_dict,
                                             regex_list=self.database.full_regex_list)

        argentinian_postcodes_match = partial(post_code_match.find_postcode_town_matches,
                                              postcodes_dict=self.database.argentina_dict,
                                              regex_list=self.database.argentina_regex_list,
                                              postcode_regex_structure=AR_POSTCODE_STRUCTURE)

        brazilian_postcodes_match = partial(post_code_match.find_postcode_town_matches,
                                            postcodes_dict=self.database.brazil_dict,
                                            regex_list=self.database.brazil_regex_list,
                                            postcode_regex_structure=BR_POSTCODE_STRUCTURE)

        chilean_postcodes_match = partial(post_code_match.find_postcode_town_matches,
                                          postcodes_dict=self.database.chile_dict,
                                          regex_list=self.database.chile_regex_list,
                                          postcode_regex_structure=CL_POSTCODE_STRUCTURE)

        chinese_postcodes_match = partial(post_code_match.find_postcode_town_matches,
                                          postcodes_dict=self.database.china_dict,
                                          regex_list=self.database.china_regex_list,
                                          postcode_regex_structure=CN_POSTCODE_STRUCTURE)

        irish_postcodes_match = partial(post_code_match.find_postcode_town_matches,
                                        postcodes_dict=self.database.ireland_dict,
                                        regex_list=self.database.ireland_regex_list,
                                        postcode_regex_structure=IE_POSTCODE_STRUCTURE)

        maltese_postcodes_match = partial(post_code_match.find_postcode_town_matches,
                                          postcodes_dict=self.database.malta_dict,
                                          regex_list=self.database.malta_regex_list,
                                          postcode_regex_structure=MT_POSTCODE_STRUCTURE)

        # Group results by sample
        for sample in data:
            yield ResultRunnerPostcodeMatch(
                postcode_matches=reduce(
                    post_code_match.PostcodeMatchResult.merge, [
                        almost_all_postcodes_match(text=sample),
                        argentinian_postcodes_match(text=sample),
                        brazilian_postcodes_match(text=sample),
                        chilean_postcodes_match(text=sample),
                        chinese_postcodes_match(text=sample),
                        irish_postcodes_match(text=sample),
                        maltese_postcodes_match(text=sample)
                    ]))

    logger.info("Done postcode matching")
