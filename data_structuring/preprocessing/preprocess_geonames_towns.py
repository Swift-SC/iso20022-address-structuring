import polars as pl

from data_structuring.components.data_provider import get_country_overrides
from data_structuring.config import DatabaseConfig, PreprocessCLIConfig
from data_structuring.preprocessing.preprocess_utils import ALLCOUNTRIES_COLUMN_NAMES
from data_structuring.preprocessing.preprocess_utils import get_country_data, filter_and_save_alternate_names, \
    get_country_languages

if __name__ == "__main__":
    # Parse CLI args
    cli_args = PreprocessCLIConfig()
    # Load database config
    config = DatabaseConfig()

    # Get list of feature codes to keep
    feature_filter = list(config.geonames_town_feature_code_filter.keys())

    # Get list of countries that override the normal filtering parameters
    country_override = get_country_overrides(config=config)

    # Get dictionary of languages per country
    languages_dict = get_country_languages(
        df_country_data=get_country_data(
            country_info_path=cli_args.input_geonames_country_info_path,
            additional_languages=cli_args.geonames_additional_languages))
    # Considering the performance cost of using all domestic languages,
    # default is to override this to limit number of possibilities i.e.: English only
    languages_dict = {key: cli_args.geonames_additional_languages for key in languages_dict.keys()}

    # Convert downloaded geonames file to parquet
    df_geonames_towns = (
        pl.scan_csv(
            cli_args.input_geonames_all_countries_path,
            new_columns=ALLCOUNTRIES_COLUMN_NAMES,
            has_header=False,
            separator="\t",
            infer_schema=False,
            quote_char=None)
        .with_columns(
            pl.col('geonameid').str.to_integer(),
            # Fill null values for population
            pl.col('population').str.to_integer().fill_null(0),
            # Map feature codes to their respective minimum population thresholds
            feature_code_min=pl.col('feature code').replace_strict(
                old=list(config.geonames_town_feature_code_filter.keys()),
                new=list(config.geonames_town_feature_code_filter.values()),
                default=config.town_minimal_population))
        .filter(
            # Remove rows with no valid country code
            (~pl.col('country code').is_null()),
            # Keep if row is defining a town/village
            (pl.col("feature class") == "P")
            # or if row is defining a special settlement
            | ((pl.col("feature class").is_in(["L", "S", "H", "A"]))
               & (pl.col("feature code").is_in(feature_filter))),
            # Filter the geonames dataframe to include only towns with a population greater than or equal
            # to the specified minimum population threshold, or towns within the specified country override.
            ((pl.col('feature code').is_in(config.geonames_country_override_code_filter))
             & (pl.col('country code').is_in(country_override)))
            | (pl.col('population') >= pl.col('feature_code_min')))
    )

    df_geonames_towns.sink_parquet(config.geonames_parquet)

    # Generate aliases for towns
    df_geonames_towns_filtered = (
        df_geonames_towns
        .select('geonameid', 'name', 'country code', 'cc2')
        .with_columns(
            # Collecting all country codes for each town based on its country
            cc_list=pl.col('country code').list.concat(pl.col('cc2').fill_null("").str.split(','))
            .list.unique().list.filter(pl.element() != ""))
        .drop('country code', 'cc2')
        .explode('cc_list')
    )

    filter_and_save_alternate_names(
        alternate_names_path=cli_args.input_geonames_alternate_names_path,
        df_base=df_geonames_towns_filtered,
        key_column='name',
        name_column='name',
        languages=languages_dict,
        default_languages=cli_args.geonames_additional_languages,
        save_location=config.town_aliases
    )
