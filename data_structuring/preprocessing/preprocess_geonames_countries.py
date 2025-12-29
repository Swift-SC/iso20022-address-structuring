import polars as pl

from data_structuring.config import DatabaseConfig, PreprocessCLIConfig
from data_structuring.preprocessing.preprocess_utils import filter_and_save_alternate_names, get_country_languages, \
    ALLCOUNTRIES_COLUMN_NAMES
from data_structuring.preprocessing.preprocess_utils import get_country_data

if __name__ == "__main__":
    # Parse CLI args
    cli_args = PreprocessCLIConfig()
    # Load database config
    config = DatabaseConfig()

    # Get list of feature codes to keep
    feature_filter = list(config.geonames_country_feature_code_filter.keys())

    # Process alternate names for countries
    df_geonames_countries = (
        get_country_data(
            country_info_path=cli_args.input_geonames_country_info_path,
            additional_languages=cli_args.geonames_additional_languages))

    # Get dictionary of languages per country
    languages_dict = get_country_languages(df_country_data=df_geonames_countries)

    df_countries = (
        # Add the list of country names
        df_geonames_countries
        .select(pl.all().exclude('Languages'))
        .with_columns(cc_list=pl.concat_list(pl.col('ISO')))
        .explode('cc_list'))

    filter_and_save_alternate_names(
        alternate_names_path=cli_args.input_geonames_alternate_names_path,
        df_base=df_countries,
        key_column='ISO',
        name_column='Country',
        languages=languages_dict,
        default_languages=cli_args.geonames_additional_languages,
        save_location=config.country_aliases,
        partition_by_key=True
    )

    # Considering the performance cost of using all domestic languages,
    # default is to override this to limit number of possibilities i.e.: English only
    languages_dict = {key: cli_args.geonames_additional_languages for key in languages_dict.keys()}

    # Generate aliases for countries
    df_countries_divisions = (
        pl.scan_csv(cli_args.input_geonames_all_countries_path,
                    new_columns=ALLCOUNTRIES_COLUMN_NAMES,
                    has_header=False,
                    separator='\t',
                    infer_schema=False,
                    quote_char=None)
        .with_columns(
            pl.col('geonameid').str.to_integer(),
            # Fill null values for population
            pl.col('population').str.to_integer().fill_null(0),
            # Map feature codes to their respective minimum population thresholds
            feature_code_min=pl.col('feature code').replace_strict(
                old=list(config.geonames_country_feature_code_filter.keys()),
                new=list(config.geonames_country_feature_code_filter.values()),
                default=config.town_minimal_population))
        .filter(
            # Remove rows with no valid country code
            (~pl.col('country code').is_null()),
            # Filter the geonames dataframe to include only provinces with a population
            # greater than or equal to the specified minimum population threshold.
            (pl.col("feature code").is_in(feature_filter))
            & (pl.col('population') >= pl.col('feature_code_min')))
        .with_columns(
            # Collect all country codes per country
            cc_list=pl.col('country code').list.concat(pl.col('cc2').fill_null("").str.split(','))
            .list.unique().list.filter(pl.element() != ""))
        .select('geonameid', 'cc_list', 'name')
        .rename({'cc_list': 'ISO', 'name': 'Country'})
        .explode('ISO')
        .with_columns(cc_list=pl.concat_list(pl.col('ISO')))
        .explode('cc_list'))

    filter_and_save_alternate_names(
        alternate_names_path=cli_args.input_geonames_alternate_names_path,
        df_base=df_countries_divisions,
        key_column='ISO',
        name_column='Country',
        languages=languages_dict,
        default_languages=cli_args.geonames_additional_languages,
        save_location=config.country_province_aliases,
        partition_by_key=True
    )
