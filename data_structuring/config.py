import itertools
import os
import string
from abc import ABC
from importlib import resources
from pathlib import Path
from typing import Literal, Any

from pydantic import field_validator, ValidationInfo, Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict, CliImplicitFlag

import data_structuring
from data_structuring.components.tags import Tag, BIOTag


class BaseSettingsISO(BaseSettings, ABC):
    # Set prefix
    model_config = SettingsConfigDict(env_prefix='ds_', cli_parse_args=True, cli_ignore_unknown_args=True)


class DatabaseConfig(BaseSettingsISO):
    # Path to the subfolder where all data files can be found
    # IMPORT: the prefix should be the FIRST field of this class (refer to the validation below)
    prefix_folder_path: Path = resources.files(data_structuring.__name__) / ".." / "resources"

    # ############################################################################################
    # GeoNames sources
    # ############################################################################################

    # GeoNames parquet file
    geonames_parquet: Path = "towns_all_countries.parquet"

    # file containing town aliases
    town_aliases: Path = "town_aliases.json"

    # file containing country aliases
    country_aliases: Path = "country_names.json"

    # file containing country aliases
    country_province_aliases: Path = "country_province_names.json"

    # GeoNames postcodes
    postcodes_almost_all_countries_dict: Path = "post_codes/almost_all_countries_dict.json"
    postcodes_almost_all_countries_regex_list: Path = "post_codes/almost_all_countries_regex_list.json"

    postcodes_argentinian_dict: Path = "post_codes/argentina_dict.json"
    postcodes_argentinian_regex_list: Path = "post_codes/argentina_regex_list.json"

    postcodes_brazilian_dict: Path = "post_codes/brazil_dict.json"
    postcodes_brazilian_regex_list: Path = "post_codes/brazil_regex_list.json"

    postcodes_chilean_dict: Path = "post_codes/chile_dict.json"
    postcodes_chilean_regex_list: Path = "post_codes/chile_regex_list.json"

    postcodes_chinese_dict: Path = "post_codes/china_dict.json"
    postcodes_chinese_regex_list: Path = "post_codes/china_regex_list.json"

    postcodes_ireland_dict: Path = "post_codes/ireland_dict.json"
    postcodes_ireland_regex_list: Path = "post_codes/ireland_regex_list.json"

    postcodes_maltese_dict: Path = "post_codes/malta_dict.json"
    postcodes_maltese_regex_list: Path = "post_codes/malta_regex_list.json"

    # ############################################################################################
    # OpenStreetMap source
    # ############################################################################################

    # OSM parquet file
    enable_osm_data: bool = False
    town_entities_osm: Path = "cities_osm_cleaned.parquet"

    # ############################################################################################
    # Other sources
    # ############################################################################################

    # file containing countries and towns that use the same name
    country_town_same_name: Path = "misc/country_city_same_name.json"

    # file containing country groupings
    country_groupings: Path = "misc/country_groupings_with_iso_code.json"

    # file containing other country identifying data points
    country_specs: Path = "misc/country_specs.json"

    # data settings
    town_minimal_population: int = 500
    force_countries: list[str] = []
    force_country_groupings: list[str] = []
    geonames_country_override_code_filter: list[str] = ["PPLA"]
    geonames_town_feature_code_filter: dict[str, int] = {
        "PPLA": town_minimal_population,
        "PPLC": town_minimal_population,
        "PPL": town_minimal_population,
        "PPLX": 0,
        "INDS": 0,
        "LTER": 0,
        "HBR": 0,
        "PRT": 0,
        "AIRP": 0,
        "MILB": 0,
        "NVB": 0,
    }
    geonames_country_feature_code_filter: dict[str, int] = {
        "ADM1": 0,
        "ISL": town_minimal_population,
    }

    # Validator to dynamically add prefix to paths
    @field_validator("*", mode="after")
    @classmethod
    def prepend_prefix(cls, p, info: ValidationInfo):

        def process_field(parameter, info_dict):
            # Only process paths, ignore other types
            if not isinstance(parameter, Path):
                return parameter

            # Do not prepend prefix to the prefix itself
            if "prefix_folder_path" not in info_dict.data:
                return parameter

            # Prepend prefix to all other fields
            return info_dict.data["prefix_folder_path"] / parameter

        # if list of parameters
        if isinstance(p, list):
            return [process_field(e, info) for e in p]
        return process_field(p, info)


class PostProcessingConfig(BaseSettingsISO):
    # Thresholding : matches below this final score will never be picked as final, not even shown
    # Score is between 0 and 1, soft capped at 1 for now
    # Optimal threshold is probably higher, but 15% is the safe and tested option
    minimal_final_score_country: float = 0.15
    minimal_final_score_town: float = 0.15
    iban_pattern: str = r"(?=([A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){1,7}))"

    # Base score to be applied on the suggested country
    # if the forced_suggested_country flag is not set
    base_score_suggested_country: float = 0.95

    # List of countries that typically include their provinces in their addresses
    # and frequently use the abbreviations for these provinces
    countries_with_common_provinces: list[str] = ['CN', 'US']

    # Flags config
    is_metropolis_threshold: int = 1_000_000
    is_small_town_threshold: int = 12_000
    part_of_street_ratio: float = 0.50
    show_inferred_country: bool = True

    # Missing entry multipliers : defines malus multipliers that compensate for the bonus of flags
    # that rely on the presence of the corresponding country or town when
    # the combination is lacking either a country or a town
    no_town_found_mul: float = 0.7  # country found but not town
    no_country_found_mul: float = 0.1  # town found but not country


class PostProcessingTownWeightsConfig(BaseSettingsISO):
    # Town weights config
    # Bonuses
    is_in_last_third: float = 0.01
    could_be_reasonable_mistake: float = 0.25
    country_is_present_bonus: float = 0.15
    suggested_country_is_present_bonus: float = 0.65
    mlp_country_is_present_bonus: float = 0.05
    is_very_close_to_country: float = 0.30
    is_on_same_line_as_country: float = 0.15
    postcode_for_town_found: float = 0.4
    is_metropolis: float = 0.10
    is_alone_on_line: float = 0.20
    # Maluses (must be negative)
    contains_typo: float = -0.85
    is_inside_another_word: float = -0.65
    is_in_first_third: float = -0.01
    is_short: float = -0.25
    is_inside_another_lower_ranked_match: float = -0.30
    is_small_town: float = -0.20
    is_small_town_and_country_not_present: float = -0.30
    country_is_present_malus: float = 0.10
    is_from_extended_data: float = -0.15
    is_not_largest_town_with_name: float = -0.10
    is_inside_street: float = -0.20
    is_common_state_province_alias: float = -0.10
    is_uncommon_state_province_alias: float = -0.15
    is_short_and_nonzero_dist_score: float = -2.00
    is_short_and_is_inside_another_word: float = -2.00
    is_inside_another_higher_ranked_match: float = -2.00


class PostProcessingCountryWeightsConfig(BaseSettingsISO):
    # Country weights config
    # Bonuses
    is_in_last_third: float = 0.01
    could_be_reasonable_mistake: float = 0.10
    town_is_present: float = 0.20
    is_very_close_to_town: float = 0.20
    is_on_same_line_as_town: float = 0.10
    postal_code_is_present: float = 0.10
    iban_is_present: float = 0.10
    phone_prefix_is_present: float = 0.10
    domain_is_present: float = 0.10
    mlp_strongly_agrees: float = 0.20
    mlp_agrees: float = 0.15
    mlp_doesnt_disagree: float = 0.05
    # Maluses (must be negative)
    contains_typo: float = -0.50
    is_inside_another_word: float = -0.60
    is_in_first_third: float = -0.01
    is_short: float = -0.05
    is_inside_another_lower_ranked_match: float = -0.30
    is_inside_street: float = -0.20
    is_common_state_province_alias: float = -0.10
    is_uncommon_state_province_alias: float = -0.15
    is_short_and_nonzero_dist_score: float = -2.00
    is_short_and_is_inside_another_word: float = -2.00
    is_inside_another_higher_ranked_match: float = -2.00


class FuzzyMatchConfig(BaseSettingsISO):
    num_workers: int = os.cpu_count() - 1

    # How many errors are allowed in a match ? If 0, this is equivalent to exact matching
    fuzzy_match_score_cutoff: int = 80
    fuzzy_match_tolerance: int = 1

    fuzzy_match_score_cutoff_towns: int = 80
    fuzzy_match_tolerance_towns: int = 1


class CRFConfig(BaseSettingsISO):
    # Model and device to use at inference time
    device: str = "cpu"
    model_weights_path: Path = (resources.files(data_structuring.__name__) / ".." / "resources"
                                / "models" / "CRF_with_MLP_EPOCH_1.safetensors")
    model_config_path: Path = (resources.files(data_structuring.__name__) / ".." / "resources"
                               / "models" / "CRF_with_MLP_EPOCH_1.config.json")

    # Only use allowed SWIFT X charset
    vocabulary: list[str] = (list(string.ascii_uppercase)
                             + list(string.digits)
                             + list(r"/-?:().,'+ {}")
                             + ["\r", "\n"]
                             )

    # Model parameters
    embedding_dimension: int = 128
    n_heads: int = 2
    depth: int = 8
    max_sequence_length: int = 224

    tags_to_keep: list[Tag] = list(Tag)

    bio_tags_to_keep: list[BIOTag] = [BIOTag.create_other()] + list(
        itertools.chain.from_iterable(
            BIOTag.create_all(tag)
            for tag in tags_to_keep
            if tag != BIOTag.create_other().tag  # skipping OTHER tag
        )
    )

    # NOTE Use small regularisation on the emissions, as this allows the
    # transformer to "hesitate" a little between the plausible tags.
    # But zero regularisation leads to some overfitting.
    regularisation_emissions: float = 1e-3

    # CRF transition matrix regularisation
    regularisation_transitions: float = 0.1
    regularisation_transitions_order_2: float = 0.01


class RunCLIConfig(BaseSettingsISO):
    # I/O
    input_data_path: Path = Field(Path(resources.files(data_structuring.__name__)
                                       / ".." / "resources" / "input" / "addresses_gauntlet.csv"),
                                  description="Input file name",
                                  alias=AliasChoices('i', 'input_path'))
    output_data_path: Path = Field(Path("data_structuring_output.csv"),
                                   description='Output file name',
                                   alias=AliasChoices('o', 'output_path'))
    verbose: CliImplicitFlag[bool] = Field(False,
                                           description="Add details of the predictions to the output file",
                                           alias=AliasChoices('v', 'verbose'))

    # Inference parameters
    batch_size: int = 1024

    logging_config: Path | None = Field(default=None,
                                        description="Path to the logging configuration file",
                                        alias=AliasChoices('l', 'logging_config'))


class PreprocessCLIConfig(BaseSettingsISO):
    # I/O GeoNames towns and countries
    input_geonames_all_countries_path: Path = Field(
        Path(resources.files(data_structuring.__name__)
             / ".." / "resources" / "raw" / "geonames" / "allCountries.txt"),
        description="GeoNames allCountries.txt path",
        alias=AliasChoices('i_ac', 'input_geonames_all_countries_path'))

    input_geonames_alternate_names_path: Path = Field(
        Path(resources.files(data_structuring.__name__)
             / ".." / "resources" / "raw" / "geonames" / "alternateNamesV2.txt"),
        description="GeoNames alternateNamesV2.txt path",
        alias=AliasChoices('i_an', 'input_geonames_alternate_names_path'))

    input_geonames_country_info_path: Path = Field(
        Path(resources.files(data_structuring.__name__)
             / ".." / "resources" / "raw" / "geonames" / "countryInfo.txt"),
        description="GeoNames countryInfo.txt path",
        alias=AliasChoices('i_ci', 'input_geonames_country_info_path'))

    # Always add English by default
    geonames_additional_languages: list[str] = ["en"]


class PreprocessRestCountriesCLIConfig(BaseSettingsISO):
    # I/O restCountries JSON file
    input_rest_countries_path: Path = Field(
        Path(resources.files(data_structuring.__name__)
             / ".." / "resources" / "raw" / "restCountries" / "countriesV3.1.json"),
        description="restCountries countriesV3.1.json path",
        alias=AliasChoices('i_rc', 'input_restCountries_path'))


class PreprocessPostcodesCLIConfig(BaseSettingsISO):
    # I/O GeoNames postcodes
    input_geonames_postcodes_all_countries_path: Path = Field(
        Path(resources.files(data_structuring.__name__)
             / ".." / "resources" / "raw" / "postcodes" / "allCountries.txt"),
        description="GeoNames postcodes allCountries.txt path",
        alias=AliasChoices('i_ac',
                           'input_geonames_postcodes_all_countries_path'))

    input_geonames_postcodes_ca_full_path: Path = Field(
        Path(resources.files(data_structuring.__name__)
             / ".." / "resources" / "raw" / "postcodes" / "CA_full.txt"),
        description="GeoNames postcodes CA_full.txt path",
        alias=AliasChoices('i_ca', 'input_geonames_postcodes_ca_full_path'))

    input_geonames_postcodes_gb_full_path: Path = Field(
        Path(resources.files(data_structuring.__name__)
             / ".." / "resources" / "raw" / "postcodes" / "GB_full.txt"),
        description="GeoNames postcodes GB_full.txt path",
        alias=AliasChoices('i_gb', 'input_geonames_postcodes_gb_full_path'))

    input_geonames_postcodes_nl_full_path: Path = Field(
        Path(resources.files(data_structuring.__name__)
             / ".." / "resources" / "raw" / "postcodes" / "NL_full.txt"),
        description="GeoNames postcodes NL_full.txt path",
        alias=AliasChoices('i_nl', 'input_geonames_postcodes_nl_full_path'))


# logging
DEFAULT_LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)8s] - %(name)s@%(funcName)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stderr"
        },
        # uncomment the block below to log to a file and add 'file' in the list of handlers
        # "file": {
        #     "level": "INFO",
        #     "class": "logging.FileHandler",
        #     "formatter": "standard",
        #     "filename": "./data_structuring.log"
        # }
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "DEBUG"
        }
    }
}
