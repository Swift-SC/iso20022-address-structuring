import string
from pathlib import Path
from typing import Callable

import orjson
import polars as pl
import zlib

from data_structuring.components.data_provider.normalization import decode_and_clean_expr
from data_structuring.config import DatabaseConfig
from data_structuring.config import PreprocessPostcodesCLIConfig

# Set up char to regex mapping
CHAR_TO_REGEX = ({char: '[A-Z]' for char in string.ascii_uppercase}  # if any character except numbers or space
                 | {char: '[0-9]' for char in string.digits}  # if the character is a number
                 | {' ': '[- ]?'})  # if the character is a space

COLUMN_NAMES = ['country code', 'postal code', 'place name', 'admin name1', 'admin code1',
                'admin name2', 'admin code2', 'admin name3', 'admin code3',
                'latitude', 'longitude', 'accuracy', ]

# Define list of countries (via ISO code) that require special preprocessing
COUNTRIES_WITH_SPECIAL_POST_CODES = ['AR', 'BR', 'CL', 'CN', 'IE', 'MT']


def filter_and_clean_dataframe(input_df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        input_df.with_columns(
            # Cleanup post codes
            postal_code_filtered=pl.col('postal code').str.to_uppercase().str.replace_all(r'[^A-Z0-9]', ' '))
        .with_columns(
            # Cleanup town names
            place_name_filtered=(decode_and_clean_expr(pl.col('place name').str.to_uppercase())
                                 .str.replace_all(r'[^A-Z0-9]', ' ')))
        .filter(pl.col('place_name_filtered') != "")
        .with_columns(
            # Generate regex of each post code
            postal_regex=pl.col('postal_code_filtered').str.replace_many(CHAR_TO_REGEX))
        .with_columns(
            # Obtain list of country codes per filtered post code
            country_code_list=pl.col('country code').unique().implode().over(partition_by='postal_code_filtered'),
            # Obtain list of place names per filtered post code
            place_name_list=pl.col('place name').unique().implode().over(partition_by='postal_code_filtered'),
            # Obtain list of decoded place names per filtered post code
            place_name_filtered_list=(
                pl.col('place_name_filtered').unique().implode().over(partition_by='postal_code_filtered')))
        .with_columns(
            # Concatenate lists of place names and iso codes
            post_code_mapping=pl.col('country_code_list').list.unique().list.concat(['place_name_filtered_list']))
        .group_by(
            # Keep only the first of each unique decoded postal code
            by='postal_code_filtered')
        .first()
        .select('postal_code_filtered', 'post_code_mapping', 'postal_regex')
    )


def preprocess_and_save(input_df: pl.LazyFrame,
                        save_path_dict: Path | str,
                        save_path_regex_list: Path | str,
                        preprocess_func: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None):
    # if no special preprocessing is required, just return the LazyFrame
    def noop_func(arg):
        return arg

    if preprocess_func is None:
        preprocess_func = noop_func

    df = filter_and_clean_dataframe(preprocess_func(input_df)).collect(engine='streaming')

    # Save the dictionary of post codes to town aliases as a compressed JSON file
    with open(save_path_dict, "wb") as f:
        f.write(
            zlib.compress(
                orjson.dumps(
                    {k: v[0]
                     for k, v in df.select(
                        'postal_code_filtered', 'post_code_mapping'
                    ).rows_by_key(key=['postal_code_filtered'], unique=True).items()}),
                level=-1
            )
        )

    # Save the list of all post codes' regex as a compressed JSON file
    with open(save_path_regex_list, "wb") as f:
        f.write(
            zlib.compress(
                orjson.dumps(df.group_by(
                    # Keep only the first of each unique postal code regex
                    by='postal_regex'
                ).first().select('postal_regex').to_series().to_list()),
                level=-1
            )
        )


def preprocess_argentina(input_df: pl.LazyFrame) -> pl.LazyFrame:
    return input_df.with_columns(
        # Prepend state code to the argentinian post code
        (pl.col('admin code1') + pl.col('postal code')).alias('postal code')
    )


def preprocess_brazil(input_df: pl.LazyFrame) -> pl.LazyFrame:
    return input_df.with_columns(
        # Take only the first 5 characters
        pl.col('postal code').str.head(5)
    )


def preprocess_chile(input_df: pl.LazyFrame) -> pl.LazyFrame:
    return input_df.with_columns(
        # Take only the first 3 characters
        pl.col('postal code').str.head(3)
    )


def preprocess_china(input_df: pl.LazyFrame) -> pl.LazyFrame:
    return input_df.with_columns(
        # Take only the first 4 characters
        pl.col('postal code').str.head(4)
    )


def preprocess_ireland(input_df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        pl.concat(
            [input_df,
             input_df.filter(pl.col('place name').str.contains(r"(?i)Dublin"))
             .with_columns(
                 # Dublin uses multiple post codes, so these need to be duplicated to
                 # represent the different ways of writing them
                 ('D' + pl.col('postal code').str.slice(1).str.strip_chars_start("0")).alias('postal code'))]
        )
    )


if __name__ == "__main__":
    # Parse CLI args
    cli_args = PreprocessPostcodesCLIConfig()
    # Load database config
    config = DatabaseConfig()

    combined_df = pl.concat(
        [pl.scan_csv(cli_args.input_geonames_postcodes_all_countries_path,
                     new_columns=COLUMN_NAMES,
                     has_header=False,
                     separator='\t',
                     infer_schema=False)
         .filter(
            # Remove the incomplete post codes lists
            ~pl.col('country code').is_in(['GB', 'CA', 'NL']))]
        + [pl.scan_csv(file_path,
                       new_columns=COLUMN_NAMES,
                       has_header=False,
                       separator='\t',
                       infer_schema=False)
           for file_path in [cli_args.input_geonames_postcodes_ca_full_path,
                             cli_args.input_geonames_postcodes_gb_full_path,
                             cli_args.input_geonames_postcodes_nl_full_path]])

    all_countries_df = combined_df.lazy().filter(
        # Remove any rows with no town names
        ~pl.all_horizontal(pl.col('place name').is_null()))

    # Save almost all countries
    print('Generating almost_all_countries postcodes files')
    preprocess_and_save(
        input_df=all_countries_df.filter(~pl.col('country code').is_in(COUNTRIES_WITH_SPECIAL_POST_CODES)),
        save_path_dict=config.postcodes_almost_all_countries_dict,
        save_path_regex_list=config.postcodes_almost_all_countries_regex_list,
    )

    # Save Argentina-specific dataset
    print('Generating argentinian postcodes files')
    preprocess_and_save(
        input_df=all_countries_df.filter(pl.col('country code') == 'AR'),
        save_path_dict=config.postcodes_argentinian_dict,
        save_path_regex_list=config.postcodes_argentinian_regex_list,
        preprocess_func=preprocess_argentina
    )

    # Save Brazil-specific dataset
    print('Generating brazilian postcodes files')
    preprocess_and_save(
        input_df=all_countries_df.filter(pl.col('country code') == 'BR'),
        save_path_dict=config.postcodes_brazilian_dict,
        save_path_regex_list=config.postcodes_brazilian_regex_list,
        preprocess_func=preprocess_brazil
    )

    # Save Chile-specific dataset
    print('Generating chilean postcodes files')
    preprocess_and_save(
        input_df=all_countries_df.filter(pl.col('country code') == 'CL'),
        save_path_dict=config.postcodes_chilean_dict,
        save_path_regex_list=config.postcodes_chilean_regex_list,
        preprocess_func=preprocess_chile
    )

    # Save China-specific dataset
    print('Generating chinese postcodes files')
    preprocess_and_save(
        input_df=all_countries_df.filter(pl.col('country code') == 'CN'),
        save_path_dict=config.postcodes_chinese_dict,
        save_path_regex_list=config.postcodes_chinese_regex_list,
        preprocess_func=preprocess_china
    )

    # Save Ireland-specific dataset
    print('Generating irish postcodes files')
    preprocess_and_save(
        input_df=all_countries_df.filter(pl.col('country code') == 'IE'),
        save_path_dict=config.postcodes_ireland_dict,
        save_path_regex_list=config.postcodes_ireland_regex_list,
        preprocess_func=preprocess_ireland
    )

    # Save Malta-specific dataset
    print('Generating maltese postcodes files')
    preprocess_and_save(
        input_df=all_countries_df.filter(pl.col('country code') == 'MT'),
        save_path_dict=config.postcodes_maltese_dict,
        save_path_regex_list=config.postcodes_maltese_regex_list,
    )
