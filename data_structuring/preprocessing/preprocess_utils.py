import io
from pathlib import Path

import orjson
import polars as pl
import zlib

from data_structuring.components.data_provider.normalization import decode_and_clean_expr

ALLCOUNTRIES_COLUMN_NAMES = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
                             'feature class',
                             'feature code', 'country code', 'cc2', 'admin1 code', 'admin2 code', 'admin3 code',
                             'admin4 code',
                             'population', 'elevation', 'dem', 'timezone', 'modification date']

COUNTRY_INFO_COLUMN_NAMES = ['ISO', 'ISO3', 'ISO-Numeric', 'fips', 'Country', 'Capital', 'Area(in sq km)', 'Population',
                             'Continent', 'tld', 'CurrencyCode', 'CurrencyName', 'Phone', 'Postal Code Format',
                             'Postal Code Regex', 'Languages', 'geonameid', 'neighbours', 'EquivalentFipsCode']

ALTERNATE_NAMES_COLUMN_NAMES = ['alternateNameId', 'geonameid', 'isolanguage', 'alternate name',
                                'isPreferredName', 'isShortName', 'isColloquial', 'isHistoric', 'from', 'to']

ALTERNATE_NAMES_LANG_COLUMN_NULL_VALUE = "special"


def filter_and_save_alternate_names(alternate_names_path: Path,
                                    df_base: pl.LazyFrame,
                                    key_column: str,
                                    name_column: str,
                                    languages: dict[str, list[str]],
                                    default_languages: list[str],
                                    save_location: Path | str,
                                    partition_by_key: bool = False) -> None:
    # Cleanup all names in the base dataset
    df_base = df_base.with_columns(decode_and_clean_expr(pl.col(name_column)).str.to_uppercase())
    # Determine partitioning column for collecting alternate names
    partitioning_col = key_column if partition_by_key else name_column
    with open(save_location, "wb") as f:
        f.write(
            zlib.compress(
                orjson.dumps({
                    town: aliases[0]
                    for town, aliases in
                    pl.scan_csv(
                        # Convert downloaded geonames alternate names file
                        alternate_names_path,
                        new_columns=ALTERNATE_NAMES_COLUMN_NAMES,
                        has_header=False,
                        separator='\t',
                        infer_schema=False,
                        quote_char=None)
                    .with_columns(
                        pl.col('geonameid').str.to_integer(),
                        # Fill null values for languages
                        pl.col('isolanguage').fill_null(ALTERNATE_NAMES_LANG_COLUMN_NULL_VALUE),
                        # Cleanup isColloquial & isHistoric columns to be binary
                        pl.col('isColloquial').replace_strict(old=["0", "1"], new=[False, True], default=False),
                        pl.col('isHistoric').replace_strict(old=["0", "1"], new=[False, True], default=False))
                    .filter(
                        # Remove irrelevant alternate names
                        ~pl.col('isColloquial'), ~pl.col('isHistoric'))
                    .select('geonameid', 'alternate name', 'isolanguage')
                    .join(
                        # Right-join with provided dataset
                        other=df_base,
                        on='geonameid',
                        how='right',
                        maintain_order=None)
                    .with_columns(
                        # Use list of country codes to determine languages
                        pl.col('cc_list').replace_strict(old=languages, default=default_languages))
                    .filter(
                        # Remove alternate names that are not in the list of languages
                        # and keep rows without an alternate name
                        ((~pl.col('alternate name').is_null()) & (pl.col('isolanguage').is_in(pl.col('cc_list'))))
                        | (pl.col('alternate name').is_null()))
                    .with_columns(
                        # Cleanup the remaining alternate names
                        alternate_name=decode_and_clean_expr(
                            pl.col('alternate name')).str.strip_chars().str.to_uppercase().fill_null(""))
                    .with_columns(
                        # Collect all alternate names using the assigned partitioning column
                        alternate_names=(
                            pl.concat_list(
                                pl.col('alternate_name').implode().over(
                                    partition_by=partitioning_col).list.unique(),
                                pl.col(name_column))
                            .list.unique().list.filter(pl.element() != "")))
                    .group_by(by=key_column)
                    .first()
                    .select(key_column, 'alternate_names')
                    .collect(engine="streaming")
                    .rows_by_key(key_column, unique=True)
                    .items()
                }),
                level=1
            )
        )


def get_country_data(country_info_path: Path, additional_languages: list[str]) -> pl.LazyFrame:
    # Always add abbreviations by default
    additional_languages += ['abbr']
    with open(country_info_path, "r") as f:
        # Obtain domestic languages for each country
        return (
            pl.scan_csv(
                io.StringIO(
                    '\n'.join([line
                               for line in f.read().split('\n')
                               if not line.startswith('#')])),
                new_columns=COUNTRY_INFO_COLUMN_NAMES,
                has_header=False,
                separator='\t',
                infer_schema=False,
                quote_char=None)
            .select('geonameid', 'ISO', 'Country', 'Languages')
            .with_columns(
                pl.col('geonameid').str.to_integer(),
                pl.col('Languages').str.split(','))
            .explode('Languages')
            .group_by(['geonameid', 'ISO', 'Country', 'Languages'])
            .agg(
                # Aggregate to keep a single list of all domestic languages
                # and duplicate country-specific languages to keep both
                # general language and country-specific language
                pl.col('Languages').str.split('-').list.get(0).alias('lang1'),
                pl.col('Languages').alias('lang2'),
                pl.all().exclude('Languages'))
            .with_columns(
                # Concat both aggregated columns into a singular column
                pl.col('lang1').list.concat(pl.col('lang2')).list.unique().alias('Languages'))
            .drop('lang1', 'lang2')
            .explode('Languages')
            .group_by(['geonameid', 'ISO', 'Country'])
            .agg(
                pl.concat_list(pl.col('Languages').implode(), additional_languages).list.unique().alias('Languages'),
                pl.all().exclude('Languages'))
        )


def get_country_languages(df_country_data: pl.LazyFrame) -> dict[str, list[str]]:
    return {
        iso: languages[0]
        for iso, languages in
        df_country_data.select('ISO', 'Languages').collect(engine="streaming")
        .rows_by_key('ISO', unique=True)
        .items()
    }
