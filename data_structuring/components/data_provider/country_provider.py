"""
Provides helper function to load country data
"""
import logging
from collections import defaultdict
from typing import Any

import orjson
import zlib

from data_structuring.components.data_provider.normalization import decode_and_clean_str
from data_structuring.config import DatabaseConfig

logger = logging.getLogger(__name__)


def load_country_dict(config: DatabaseConfig,
                      country_provinces: dict) -> tuple[dict[Any, Any], dict[str, Any]]:
    """
    Loads a dictionary of country names and their aliases.

    Args:
        config: a DatabaseConfig object the contains the path to the country dictionary mapping
          country names to their different spelling variations
        country_provinces: list of dictionaries mapping region name to its country name

    Returns:
        a dictionary that maps all name variations for all countries (keys) to their ISO2 code (value)
    """
    logger.info("Loading country aliases from %s", config.country_aliases)
    with open(config.country_aliases, "rb") as file:
        countries = orjson.loads(zlib.decompress(file.read()))

    alpha2_to_alpha2 = {k.casefold(): k for k in countries.keys()}
    unwrapped_countries = defaultdict(list[str])

    for cc, alias_list in countries.items():
        for alias in alias_list:
            unwrapped_countries[decode_and_clean_str(alias).casefold()].append(cc)

    for cc, provinces in country_provinces.items():
        for province in provinces:
            unwrapped_countries[decode_and_clean_str(province).casefold()].append(cc)

    unwrapped_countries[decode_and_clean_str("NO COUNTRY").casefold()] = ["NO COUNTRY"]
    logger.info("Loaded %i country aliases", len(unwrapped_countries))
    return alpha2_to_alpha2, unwrapped_countries


def get_country_overrides(config: DatabaseConfig) -> list[str]:
    """
    Returns the list of country overrides from the configuration (groups and single countries).
    """

    overrides: list[str] = []
    if len(config.force_country_groupings):
        logger.info("Loading country groupings from %s", config.country_groupings)
        with open(config.country_groupings) as f:
            country_groupings = orjson.loads(f.read())
            overrides = list(set(
                [country for grouping in config.force_country_groupings
                    if grouping in country_groupings
                    for country in country_groupings[grouping]] + config.force_countries))
        logger.info("Loaded %i countries from %i groups",
                    len(overrides),
                    len(country_groupings))
    else:
        logger.info("country_groupings is empty. Bypassing %s", config.country_groupings)
        overrides = config.force_countries
        logger.info("Loaded %i countries", len(overrides))

    return overrides
