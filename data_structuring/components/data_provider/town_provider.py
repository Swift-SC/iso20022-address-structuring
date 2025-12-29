"""
Helper functions to load town data
"""
import orjson
import logging

from data_structuring.components.data_provider.normalization import decode_and_clean_str
from data_structuring.config import DatabaseConfig

logger = logging.getLogger(__name__)


def load_countries_towns_with_same_name(config: DatabaseConfig,
                                        all_possibilities_town: dict[str, set[str]],
                                        largest_country_code_for_town: dict[str, str]):
    """
    Loads countries towns with the same name as their countries and update the dictionaries given in parameter
    Args:
        config: configuration object
        all_possibilities_town: The mapping of town names to list of country codes
        largest_country_code_for_town: mapping of town names to country code of most populous instance

    Returns:
        a dictionary mapping town names to ISO2 their iso country codes
    """
    logger.info("Loading towns with same name than their country from %s", config.country_town_same_name)
    with open(config.country_town_same_name) as f:
        country_town_same_name = orjson.loads(f.read())

        for name, iso in country_town_same_name.items():
            key = decode_and_clean_str(name.casefold())
            all_possibilities_town[key].add(iso)

            if key not in largest_country_code_for_town:
                largest_country_code_for_town[key] = iso

    logger.info("Loaded %i towns", len(country_town_same_name))
    return country_town_same_name
