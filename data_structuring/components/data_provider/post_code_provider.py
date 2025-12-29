"""
Provides helper function to load postcodes from files
"""
import logging

import orjson
import zlib

from data_structuring.config import DatabaseConfig

logger = logging.getLogger(__name__)


def load_postcode_data(config: DatabaseConfig):
    """
    Load all postcode-related data and return a tuple of all specific postcode dictionaries and regex lists
    Args:
        config: configuration object
    Returns:
        tuple of all specific postcode dictionaries and regex lists
    """

    def load_file(file_path):
        with open(file_path, "rb") as file:
            return orjson.loads(zlib.decompress(file.read()))

    full_dict = load_file(config.postcodes_almost_all_countries_dict)
    full_regex_list = load_file(config.postcodes_almost_all_countries_regex_list)
    logger.info("Loaded %i postcodes for almost all countries", len(full_dict))

    ireland_dict = load_file(config.postcodes_ireland_dict)
    ireland_regex_list = load_file(config.postcodes_ireland_regex_list)
    logger.info("Loaded %i postcodes and %i regex for IRELAND",
                len(ireland_dict),
                len(ireland_regex_list))

    malta_dict = load_file(config.postcodes_maltese_dict)
    malta_regex_list = load_file(config.postcodes_maltese_regex_list)
    logger.info("Loaded %i postcodes and %i regex for MALTA",
                len(malta_dict),
                len(malta_regex_list))

    chile_dict = load_file(config.postcodes_chilean_dict)
    chile_regex_list = load_file(config.postcodes_chilean_regex_list)
    logger.info("Loaded %i postcodes and %i regex for CHILE",
                len(chile_dict),
                len(chile_regex_list))

    argentina_dict = load_file(config.postcodes_argentinian_dict)
    argentina_regex_list = load_file(config.postcodes_argentinian_regex_list)
    logger.info("Loaded %i postcodes and %i regex for ARGENTINA",
                len(argentina_dict),
                len(argentina_regex_list))

    brazil_dict = load_file(config.postcodes_brazilian_dict)
    brazil_regex_list = load_file(config.postcodes_brazilian_regex_list)
    logger.info("Loaded %i postcodes and %i regex for BRAZIL",
                len(brazil_dict),
                len(brazil_regex_list))

    china_dict = load_file(config.postcodes_chinese_dict)
    china_regex_list = load_file(config.postcodes_chinese_regex_list)
    logger.info("Loaded %i postcodes and %i regex for CHINA",
                len(china_dict),
                len(china_regex_list))

    return (full_dict, full_regex_list,
            ireland_dict, ireland_regex_list,
            malta_dict, malta_regex_list,
            chile_dict, chile_regex_list,
            argentina_dict, argentina_regex_list,
            brazil_dict, brazil_regex_list,
            china_dict, china_regex_list)
