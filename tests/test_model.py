from importlib import resources
from pathlib import Path

import polars as pl
import pytest

import data_structuring
from data_structuring.components.readers.dataframe_reader import DataFrameReader
from data_structuring.pipeline import AddressStructuringPipeline


@pytest.fixture(autouse=True)
def gauntlet_path():
    return Path(resources.files(data_structuring.__name__) / ".." / "resources" / "input" / "addresses_gauntlet.csv")
    # return Path(resources.files(data_structuring.__name__) / ".." / "resources" / "input" / "addresses_wikipedia.csv")


@pytest.fixture(autouse=True)
def batch_size():
    return 1024


def test_gauntlet(gauntlet_path: str, batch_size: int):
    # Parse gauntlet
    df = (
        pl.read_csv(gauntlet_path, infer_schema=False)
        .with_columns(
            pl.col('town').fill_null("NO TOWN"),
            pl.col('country').fill_null("NO COUNTRY"))
        .select("address", "country", "town")
    )

    reader = DataFrameReader(df, "address")
    towns = df["town"].to_list()
    countries = df["country"].to_list()

    # Start inference
    ds = AddressStructuringPipeline()
    results = ds.run(reader, batch_size=batch_size)

    rows = []
    for result, gt_country_code, town in zip(results, countries, towns):
        prediction_country, confidence_country, ignored = result.i_th_best_match_country(0, value_if_none="NO COUNTRY")
        prediction_town, confidence_town, ignored = result.i_th_best_match_town(0, value_if_none="NO TOWN")

        rows.append({'pred_country': prediction_country, 'pred_town': prediction_town})

    df = (
        pl.concat([df, pl.DataFrame(rows)], how="horizontal")
        .with_columns(
            is_no_country=(pl.col('country') == pl.lit("NO COUNTRY")),
            is_no_town=(pl.col('town') == pl.lit("NO TOWN")),
            is_correct_country=(pl.col('country') == pl.col('pred_country')),
            is_correct_town=(pl.col('town') == pl.col('pred_town')))
    )

    n_countries = len(df.filter(~pl.col('is_no_country')))
    n_towns = len(df.filter(~pl.col('is_no_town')))

    n_no_countries = len(df.filter(pl.col('is_no_country')))
    n_no_towns = len(df.filter(pl.col('is_no_town')))

    n_correct_countries = len(df.filter((~pl.col('is_no_country')) & (pl.col('is_correct_country'))))
    n_correct_towns = len(df.filter((~pl.col('is_no_town')) & (pl.col('is_correct_town'))))

    n_correct_no_countries = len(df.filter((pl.col('is_no_country')) & (pl.col('is_correct_country'))))
    n_correct_no_towns = len(df.filter((pl.col('is_no_town')) & (pl.col('is_correct_town'))))

    n_gt_match_countries = len(df.filter(pl.col('is_correct_country')))
    n_gt_match_towns = len(df.filter(pl.col('is_correct_town')))

    n_correct_all = len(df.filter((pl.col('is_correct_town')) & (pl.col('is_correct_country'))))

    # Convert to accuracy
    n_correct_countries /= len(df)
    n_correct_towns /= len(df)
    n_correct_no_countries /= len(df)
    n_correct_no_towns /= len(df)
    n_gt_match_countries /= len(df)
    n_gt_match_towns /= len(df)
    n_correct_all /= len(df)

    print({
        # General accuracy
        "General country accuracy": n_gt_match_countries,
        "General town accuracy": n_gt_match_towns,
        "Combined general accuracy": n_correct_all,
        # Specific accuracy scores
        "Correct country (present) accuracy": n_correct_countries,
        "Correct town (present) accuracy": n_correct_towns,
        "Correct country (not present) accuracy": n_correct_no_countries,
        "Correct town (not present) accuracy": n_correct_no_towns,
        # Statistics about the dataset
        "Number of countries (present)": n_countries,
        "Number of towns (present)": n_towns,
        "Number of countries (not present)": n_no_countries,
        "Number of towns (not present)": n_no_towns
    }, flush=True)
