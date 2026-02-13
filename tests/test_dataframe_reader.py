import polars as pl
import pytest

from data_structuring.components.readers.base_reader import DEFAULT_SUGGESTED_COUNTRY_COLUMN, DEFAULT_ADDRESS_COLUMN, \
    DEFAULT_FORCE_SUGGESTED_COUNTRY_COLUMN
from data_structuring.components.readers.dataframe_reader import DataFrameReader


def test_dataframe_reader_basic():
    """Test DataFrameReader with basic DataFrame."""
    # Setup
    df = pl.DataFrame({
        DEFAULT_ADDRESS_COLUMN: ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
        "city": ["New York", "Los Angeles", "Chicago"],
        DEFAULT_SUGGESTED_COUNTRY_COLUMN: [None, "US", "US"],
        DEFAULT_FORCE_SUGGESTED_COUNTRY_COLUMN: [None, "false", "TRUE"],
    })
    # Create reader with minimal arguments
    reader = DataFrameReader(dataframe=df, data_column_name=DEFAULT_ADDRESS_COLUMN)

    # Execute without args
    results = list(reader.read())

    # Verify
    assert len(results) == 3
    assert results[0].text == "123 Main St"
    assert results[0].suggested_country is None
    assert results[0].force_suggested_country == False
    assert results[1].text == "456 Oak Ave"
    assert results[1].suggested_country == None
    assert results[1].force_suggested_country == False
    assert results[2].text == "789 Pine Rd"
    assert results[2].suggested_country == None
    assert results[2].force_suggested_country == False

    # Create reader with all arguments
    reader = DataFrameReader(
        dataframe=df,
        data_column_name=DEFAULT_ADDRESS_COLUMN,
        suggested_country_column=DEFAULT_SUGGESTED_COUNTRY_COLUMN,
        force_suggested_country_column=DEFAULT_FORCE_SUGGESTED_COUNTRY_COLUMN
    )

    # Execute
    results = list(reader.read())

    # Verify
    assert len(results) == 3
    assert results[0].text == "123 Main St"
    assert results[0].suggested_country is None
    assert results[0].force_suggested_country == False
    assert results[1].text == "456 Oak Ave"
    assert results[1].suggested_country == "US"
    assert results[1].force_suggested_country == False
    assert results[2].text == "789 Pine Rd"
    assert results[2].suggested_country == "US"
    assert results[2].force_suggested_country == True


def test_dataframe_reader_empty_dataframe():
    """Test DataFrameReader with empty DataFrame."""
    # Setup
    df = pl.DataFrame({DEFAULT_ADDRESS_COLUMN: []})
    # Create reader with minimal arguments
    reader = DataFrameReader(dataframe=df, data_column_name=DEFAULT_ADDRESS_COLUMN)

    # Execute
    results = list(reader.read())

    # Verify
    assert len(results) == 0


def test_dataframe_reader_all_null():
    """Test DataFrameReader with all null values."""
    # Setup
    df = pl.DataFrame({
        DEFAULT_ADDRESS_COLUMN: [None, None, None],
        "city": ["New York", "Los Angeles", "Chicago"],
    })
    # Create reader with minimal arguments
    reader = DataFrameReader(dataframe=df, data_column_name=DEFAULT_ADDRESS_COLUMN)

    # Execute
    results = list(reader.read())

    # Verify
    assert len(results) == 0


def test_dataframe_reader_invalid_column():
    """Test DataFrameReader raises error for invalid column name."""
    # Setup
    df = pl.DataFrame({
        DEFAULT_ADDRESS_COLUMN: ["123 Main St", "456 Oak Ave"],
        "city": ["New York", "Los Angeles"]
    })

    # Execute & Verify
    with pytest.raises(ValueError) as exc_info:
        DataFrameReader(df, "invalid_column")

    assert "Column 'invalid_column' not found in DataFrame" in str(exc_info.value)
    assert "Available columns:" in str(exc_info.value)


def test_dataframe_reader_multiple_columns():
    """Test DataFrameReader with DataFrame containing multiple columns."""
    # Setup
    df = pl.DataFrame({
        "id": [1, 2, 3],
        DEFAULT_ADDRESS_COLUMN: ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
        "city": ["New York", "Los Angeles", "Chicago"],
        "zip": ["10001", "90001", "60601"]
    })
    reader = DataFrameReader(df, "city")

    # Execute
    results = list(reader.read())

    # Verify - should only read from 'city' column
    assert len(results) == 3
    assert results[0].text == "New York"
    assert results[1].text == "Los Angeles"
    assert results[2].text == "Chicago"


def test_dataframe_reader_special_characters():
    """Test DataFrameReader handles special characters."""
    # Setup
    df = pl.DataFrame({
        DEFAULT_ADDRESS_COLUMN: [
            "123 Main St\nApt 4",
            "Café René, 456 Rue de la Paix",
            "北京市朝阳区",
            "Москва, Красная площадь"
        ]
    })
    # Create reader with minimal arguments
    reader = DataFrameReader(dataframe=df, data_column_name=DEFAULT_ADDRESS_COLUMN)

    # Execute
    results = list(reader.read())

    # Verify
    assert len(results) == 4
    assert results[0].text == "123 Main St\nApt 4"
    assert results[1].text == "Café René, 456 Rue de la Paix"
    assert results[2].text == "北京市朝阳区"
    assert results[3].text == "Москва, Красная площадь"
