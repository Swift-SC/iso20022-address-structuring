"""
Module providing towns_from_geonames function to fetch, normalize and prepare town names
"""
import logging
from collections import defaultdict

import orjson
import polars as pl
import zlib

from data_structuring.components.data_provider.normalization import (
    generate_duplicate_aliases,
    decode_and_clean_expr,
    decode_and_clean_str,
)
from data_structuring.config import DatabaseConfig

logger = logging.getLogger(__name__)


def towns_from_geonames(config: DatabaseConfig) -> tuple[dict, dict, dict]:
    """
    Reads GeoNames file to extract and organize town names and their, aliases

    Args:
        config: configuration object

    Returns:
        Tuple containing:
            - Dictionary mapping town names to list of country codes
            - Dictionary mapping town names to their population
            - Dictionary mapping town names to country code of most populous instance
    """

    logger.info("Loading towns aliases file %s", config.town_aliases)
    with open(config.town_aliases, "rb") as file:
        aliases = orjson.loads(zlib.decompress(file.read()))
    logger.info("Loaded %i town aliases", len(aliases))

    original_full_names: dict[str, set[str]] = defaultdict(set)
    towns_and_populations: dict[str, int] = defaultdict(int)
    largest_country_code_for_town: dict[str, str] = defaultdict(str)

    logger.info("Loading geonames file %s", config.geonames_parquet)
    df_geonames = (
        pl.scan_parquet(config.geonames_parquet)
        .with_columns(
            # Convert to uppercase for aliases mapping
            decoded_name=decode_and_clean_expr(pl.col('name')).str.to_uppercase())
        .with_columns(
            # Obtain list of country codes per decoded town name
            country_code_list=pl.col('country code').unique().implode().over(partition_by='decoded_name'))
        .select(
            # Keep only the highest population count for each town
            pl.all().get(0).over(partition_by='decoded_name', order_by='population', descending=True))
        # Keep only the first of each unique decoded town name
        .group_by(by='decoded_name')
        .first()
        .select('decoded_name', 'population', 'country code', 'country_code_list')
        .collect(engine="streaming")
    )

    # Process each town
    for row in df_geonames.iter_rows(named=True):
        # Generate duplicates only once
        decoded_name_duplicates = generate_duplicate_aliases(row['decoded_name'])

        # Build all possible aliases
        all_possible_aliases = {alias.casefold() for alias in decoded_name_duplicates}

        # Process aliases from dictionary lookups
        for dupe_name in decoded_name_duplicates:
            alias_names = aliases.get(dupe_name)
            if alias_names:  # Skip empty lookups
                for name in alias_names:
                    name_aliases = generate_duplicate_aliases(name)
                    all_possible_aliases.update(decode_and_clean_str(alias).casefold() for alias in name_aliases)

        # Pre-compute row values once
        country_code_set = set(row['country_code_list'])
        population = row['population']
        country_code = row['country code']

        # Process each town name (already casefolded)
        for town_name in all_possible_aliases:
            # Update population tracking if this is larger
            if population > towns_and_populations.get(town_name, 0):
                largest_country_code_for_town[town_name] = country_code
                towns_and_populations[town_name] = population

            # Update country codes for this town
            if town_name in original_full_names:
                original_full_names[town_name].update(country_code_set)
            else:
                original_full_names[town_name] = country_code_set.copy()

    # Add special keywords
    original_full_names[decode_and_clean_str("NO TOWN").casefold()] = {"NO TOWN"}

    # Log stats on the loaded dataset
    logger.info("Loaded %i original names", len(original_full_names))
    logger.info("Loaded %i towns with population", len(towns_and_populations))
    logger.info("Loaded %i largest towns to country", len(largest_country_code_for_town))

    return original_full_names, towns_and_populations, largest_country_code_for_town
