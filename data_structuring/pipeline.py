"""
This module provides an interface to running pipeline for address structuring.
"""
import logging
import warnings
from itertools import islice

from data_structuring.components.data_provider.normalization import decode_and_clean_str
from data_structuring.components.database import Database
from data_structuring.components.readers.base_reader import BaseReader, AddressSample
from data_structuring.components.runners import RunnerCRF, RunnerFuzzyMatch, RunnerPostProcessing
from data_structuring.components.runners.runner_postcode_match import RunnerPostcodeMatch
from data_structuring.config import (CRFConfig,
                                     FuzzyMatchConfig,
                                     PostProcessingConfig,
                                     DatabaseConfig,
                                     PostProcessingCountryWeightsConfig,
                                     PostProcessingTownWeightsConfig)


# Custom functions
def _batched(iterable, n):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


class AddressStructuringPipeline:
    """
    Main class of the data_structuring package.
    This class aims at encapsulating all the logic and provides the user with a clean and straightforward interface.
    """

    def __init__(self,
                 crf_config: CRFConfig | None = None,
                 fuzzy_match_config: FuzzyMatchConfig | None = None,
                 post_processing_config: PostProcessingConfig | None = None,
                 post_processing_town_weights_config: PostProcessingTownWeightsConfig | None = None,
                 post_processing_country_weights_config: PostProcessingCountryWeightsConfig | None = None,
                 database_config: DatabaseConfig | None = None):
        # Save configs or fetch default values if no explicit config is given
        self._crf_config = crf_config or CRFConfig()
        self._fuzzy_match_config = fuzzy_match_config or FuzzyMatchConfig()
        self._post_processing_config = post_processing_config or PostProcessingConfig()
        self._post_processing_town_weights_config = (post_processing_town_weights_config
                                                     or PostProcessingTownWeightsConfig())
        self._post_processing_country_weights_config = (post_processing_country_weights_config
                                                        or PostProcessingCountryWeightsConfig())
        self._database_config = database_config or DatabaseConfig()

        # Create the database
        self._database_controller = Database(config=self._database_config)

        # Create all runners
        self._crf_runner = RunnerCRF(config=self._crf_config, database=self._database_controller)
        self._fuzzy_runner = RunnerFuzzyMatch(config=self._fuzzy_match_config, database=self._database_controller)
        self._postcode_runner = RunnerPostcodeMatch(config=self._fuzzy_match_config, database=self._database_controller)
        self._post_processing_runner = RunnerPostProcessing(
            config=self._post_processing_config,
            town_weights=self._post_processing_town_weights_config,
            country_weights=self._post_processing_country_weights_config,
            database=self._database_controller)

    def _validate_sample(self, sample: str) -> True:
        """
        Raise a `ValueError` if the sample is not compliant with the current configurations, indicating what failed.
        Otherwise, return `True`.
        """

        # Check sample size
        if len(sample) > self._crf_config.max_sequence_length:
            raise ValueError(
                f"The size of the sample is {len(sample)} while the maximum sequence size is "
                f"{self._crf_config.max_sequence_length}")

        return True

    def _clean_sample(self, sample: str) -> str:
        """
        Clean the sample so that it can be processed by the pipeline
        """
        return decode_and_clean_str(sample.replace("\\n", "\n").replace("\r", "").upper())

    def _clean_and_validate_sample(self,
                                   sample: AddressSample,
                                   raise_on_validation_error: bool = True) -> AddressSample:

        try:
            self._validate_sample(sample.text)
        except ValueError as value_error:
            if raise_on_validation_error:
                raise ValueError(f"Unable to validate sample `{sample.text}`") from value_error
            else:
                warnings.warn(f"Unable to validate sample `{sample.text}`, this might lead to unexpected behaviors")

        cleaned_text = self._clean_sample(sample.text)
        return AddressSample(
            text=cleaned_text,
            suggested_country=sample.suggested_country,
            force_suggested_country=sample.force_suggested_country
        )

    def run(self, reader: BaseReader, batch_size: int = 1024):

        logger = logging.getLogger(__name__)

        # Clean samples and perform sanity checks
        samples = (self._clean_and_validate_sample(sample) for sample in reader.read())

        all_results = []

        for batch in _batched(samples, batch_size):
            # Extract text strings for runners that operate on raw text
            batch_texts = [sample.text for sample in batch]

            # CRF into fuzzy match into post-processing
            logger.info("Running CRF (1/4)")
            all_crf_results = self._crf_runner.tag(batch_texts)
            logger.info("Running Fuzzy Matching (2/4)")
            all_fuzzy_match_results = self._fuzzy_runner.match(
                batch_texts,
                suggested_countries=[sample.suggested_country for sample in batch]
            )
            logger.info("Running Postcode Matching (3/4)")
            all_postcode_match_results = self._postcode_runner.match(batch_texts)
            logger.info("Running Post Processing (4/4)")
            results = self._post_processing_runner.run(
                all_crf_results,
                all_fuzzy_match_results,
                all_postcode_match_results,
                address_samples=list(batch)
            )
            # Save current batch to the global results list
            all_results.extend(results)

        return all_results


def flatten_aliases(country_aliases: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    """Flatten {country: {town: [aliases]}} → {town: [aliases]}."""
    flat = {}
    for country, towns in country_aliases.items():
        for town, aliases in towns.items():
            flat[town] = aliases
    return flat
