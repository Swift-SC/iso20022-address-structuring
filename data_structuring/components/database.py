"""
Module responsible for loading and preparing all data needed for the runners and keep it in memory
"""
import copy
import logging

import orjson
import zlib

from data_structuring.components.data_provider import (load_country_dict,
                                                       towns_from_geonames,
                                                       load_postcode_data,
                                                       get_extended_towns,
                                                       load_countries_towns_with_same_name,
                                                       get_country_overrides)
from data_structuring.config import DatabaseConfig

logger = logging.getLogger(__name__)


class Database:
    """
    A class that loads and pre-process all the files that are necessary for the application to work.
    """
    def __init__(self, config: DatabaseConfig | None = None):
        self.config = config or DatabaseConfig()
        self._load_all_data()

    def load(self):
        self._load_all_data()
        return self

    def _load_provinces(self):
        logger.info("Loading provinces data")
        with open(self.config.country_province_aliases, "rb") as file:
            self.provinces = orjson.loads(zlib.decompress(file.read()))
        logger.info("Loaded %i provinces", len([ls for ls in self.provinces.values()]))

    def _load_all_data(self):

        self.countries_overrides = get_country_overrides(config=self.config)

        logger.info("Loading country 'ISO2 -> aliases' mapping %s", self.config.country_aliases)
        self._load_provinces()

        (self.alpha2_to_alpha2,
         self.all_possibilities_country) = load_country_dict(
            config=self.config,
            country_provinces=self.provinces
        )
        logger.info("Loaded %i iso2 codes and their aliases", len(self.alpha2_to_alpha2))

        (self.all_possibilities_town,
         self.towns_and_populations,
         self.largest_country_code_for_town) = towns_from_geonames(config=self.config)

        # Add countries with same names as towns
        self.country_town_same_name = load_countries_towns_with_same_name(
            config=self.config,
            all_possibilities_town=self.all_possibilities_town,
            largest_country_code_for_town=self.largest_country_code_for_town
        )

        (self.extended_all_possibilities_town,
         extended_towns_and_populations,
         extended_largest_country_code_for_town) = get_extended_towns(
            config=self.config,
            towns_to_remove=self.all_possibilities_town,
            country_override=self.countries_overrides
        )

        extended_towns_and_populations.update(self.towns_and_populations)
        extended_largest_country_code_for_town.update(self.largest_country_code_for_town)

        self.towns_and_populations = extended_towns_and_populations
        self.largest_country_code_for_town = extended_largest_country_code_for_town

        logger.info("Loading country specifiers from %s", self.config.country_specs)
        with open(self.config.country_specs, "rb") as f:
            self.countries_features = orjson.loads(zlib.decompress(f.read()))
        logger.info("Loaded specifiers for %i countries", len(self.countries_features))

        # Merge both towns lists for non-fuzzy matching logic
        self.full_all_possibilities_town = copy.deepcopy(self.all_possibilities_town)
        self.full_all_possibilities_town.update(
            {town: iso_list for iso, towns in self.extended_all_possibilities_town.items()
             for town, iso_list in towns.items()})

        # Load postcode data
        (self.full_dict, self.full_regex_list,
         self.ireland_dict, self.ireland_regex_list,
         self.malta_dict, self.malta_regex_list,
         self.chile_dict, self.chile_regex_list,
         self.argentina_dict, self.argentina_regex_list,
         self.brazil_dict, self.brazil_regex_list,
         self.china_dict, self.china_regex_list) = load_postcode_data(self.config)

        logger.info("All data loaded")
