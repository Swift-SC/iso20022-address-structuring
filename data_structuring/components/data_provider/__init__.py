"""
Module providing data loading and preparation functions
"""
from .country_provider import load_country_dict, get_country_overrides
from .osm_towns_provider import get_extended_towns
from .geoname_provider import towns_from_geonames
from .normalization import (
    duplicate_if_saint_in_name,
    generate_duplicate_aliases,
    duplicate_if_separator_present,
    decode_and_clean_str)
from .post_code_provider import load_postcode_data
from .town_provider import load_countries_towns_with_same_name
