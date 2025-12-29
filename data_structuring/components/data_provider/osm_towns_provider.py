"""
Provides helper function to load extended towns data from OpenStreetMap or from local file
"""
import logging
from collections import defaultdict

import polars as pl
from jellyfish import damerau_levenshtein_distance

from data_structuring.components.data_provider.normalization import generate_duplicate_aliases, \
    decode_and_clean_expr
from data_structuring.config import DatabaseConfig

APPROVED_NAME_LABELS = ['name', 'alternative_names', 'int_names', 'official_name', 'int_name', 'loc_name',
                        'short_name']
APPROVED_TOWN_LABELS = ['city', 'town', 'village']


logger = logging.getLogger(__name__)


def get_extended_towns(config: DatabaseConfig, towns_to_remove: dict, country_override: list[str] | None = None) \
        -> tuple[dict, dict, dict]:
    """
    Loads extended towns data from pre-made dictionaries if the configuration flag use_preloaded_extended_data
    is set to True or from osm data otherwise
    Args:
        config: configuration object
        towns_to_remove: list of towns to remove from final data
        country_override: list of country codes to include regardless of population

    Returns:
        Tuple containing:
            - Dictionary mapping town names to list of country codes
            - Dictionary mapping town names to their population
            - Dictionary mapping town names to country code of most populous instance
    """

    if not config.enable_osm_data:
        return {}, {}, {}
    return _load_from_osm(config, towns_to_remove, country_override)


def _load_from_osm(config: DatabaseConfig,
                   towns_to_remove: dict,
                   country_override: list[str] | None = None) -> tuple[dict, dict, dict]:
    extended_all_possibilities_town: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    extended_towns_and_populations: dict[str, int] = defaultdict(int)
    extended_largest_country_code_for_town = defaultdict(str)

    df_osm = (
        pl.scan_parquet(config.town_entities_osm)
        .with_columns(
            # Fill null values for 'NA' ISO codes (which represent an actual country)
            pl.col('iso').fill_null('NA'),
            # Fill null values for population
            pl.col('population').fill_null(0))
        .filter(
            # Filter the OSM dataframe to include only towns with a population greater than or equal
            # to the specified minimum population threshold, or towns within the specified country override,
            # and towns that have the appropriate labels
            pl.col('label').is_in(APPROVED_NAME_LABELS),
            pl.col('place_type').is_in(APPROVED_TOWN_LABELS),
            (pl.col('population') >= config.town_minimal_population) | (pl.col('iso').is_in(country_override)))
        .with_columns(
            # Decode town name using normalization function
            decoded_city_name=decode_and_clean_expr(pl.col('city_name')).map_elements(
                lambda x: x.casefold(), return_dtype=pl.String()))
        .with_columns(
            # Get the list of different ISO codes for each town
            iso_list=pl.col('iso').unique().implode().over(partition_by='decoded_city_name'))
        .select(
            # Keep only the highest population count for each town
            pl.all().get(0).over(partition_by='decoded_city_name', order_by='population', descending=True))
        .group_by(
            # Keep only the first of each unique decoded town name
            by='decoded_city_name')
        .first()
        .select('decoded_city_name', 'original_city_name', 'population', 'iso', 'iso_list')
        .filter(~pl.col('decoded_city_name').is_in(towns_to_remove))
        .with_columns(
            # Decode original town name using normalization function
            decoded_original_city_name=decode_and_clean_expr(pl.col('original_city_name')).map_elements(
                lambda x: x.casefold(), return_dtype=pl.String()))
        .with_columns(
            # Calculate damerau levenshtein distance between both town names
            decoded_city_name_distance=pl.struct('decoded_original_city_name', 'decoded_city_name').map_elements(
                lambda x: damerau_levenshtein_distance(x['decoded_original_city_name'], x['decoded_city_name']),
                return_dtype=pl.Int32()))
        .filter(
            # Keep only rows where the town names have either no or more than one differences
            pl.col('decoded_city_name_distance') != 1.0)
        .collect(engine="streaming")
    )

    for row in df_osm.iter_rows(named=True):
        # Cache values
        population = row['population']
        iso = row['iso']
        iso_list = row['iso_list']
        iso_set = set(iso_list)

        for town_name in generate_duplicate_aliases(row['decoded_city_name']):
            max_pop = max(extended_towns_and_populations[town_name], population)
            if max_pop == population:
                extended_largest_country_code_for_town[town_name] = iso
            extended_towns_and_populations[town_name] = max_pop

            # Use update() for in-place modification
            for iso_code in iso_list:
                extended_all_possibilities_town[iso_code][town_name].update(iso_set)

    # Log stats on the loaded dataset
    count = sum(len(v) for v in extended_all_possibilities_town.values())
    logger.info("Loaded %i towns", count)
    logger.info("Loaded %i towns with populations", len(extended_towns_and_populations))
    logger.info("Loaded %i country code for towns", len(extended_largest_country_code_for_town))

    return extended_all_possibilities_town, extended_towns_and_populations, extended_largest_country_code_for_town
