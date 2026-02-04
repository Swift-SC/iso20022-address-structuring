import polars as pl
import pytest

from data_structuring.components.readers.dataframe_reader import DataFrameReader


def test_dataframe_reader_basic():
    """Test DataFrameReader with basic DataFrame."""
    # Setup
    df = pl.DataFrame({
        "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
        "city": ["New York", "Los Angeles", "Chicago"]
    })
    reader = DataFrameReader(df, "address")

    # Execute
    results = list(reader.read())

    # Verify
    assert len(results) == 3
    assert results[0] == "123 Main St"
    assert results[1] == "456 Oak Ave"
    assert results[2] == "789 Pine Rd"


def test_dataframe_reader_empty_dataframe():
    """Test DataFrameReader with empty DataFrame."""
    # Setup
    df = pl.DataFrame({"address": []})
    reader = DataFrameReader(df, "address")

    # Execute
    results = list(reader.read())

    # Verify
    assert len(results) == 0


def test_dataframe_reader_all_nan():
    """Test DataFrameReader with all NaN values."""
    # Setup
    df = pl.DataFrame({
        "address": [None, None, float('nan')],
        "city": ["New York", "Los Angeles", "Chicago"]
    })
    reader = DataFrameReader(df, "address")

    # Execute
    results = list(reader.read())

    # Verify
    assert len(results) == 0


def test_dataframe_reader_invalid_column():
    """Test DataFrameReader raises error for invalid column name."""
    # Setup
    df = pl.DataFrame({
        "address": ["123 Main St", "456 Oak Ave"],
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
        "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
        "city": ["New York", "Los Angeles", "Chicago"],
        "zip": ["10001", "90001", "60601"]
    })
    reader = DataFrameReader(df, "city")

    # Execute
    results = list(reader.read())

    # Verify - should only read from 'city' column
    assert len(results) == 3
    assert results[0] == "New York"
    assert results[1] == "Los Angeles"
    assert results[2] == "Chicago"


def test_dataframe_reader_special_characters():
    """Test DataFrameReader handles special characters."""
    # Setup
    df = pl.DataFrame({
        "address": [
            "123 Main St\nApt 4",
            "Café René, 456 Rue de la Paix",
            "北京市朝阳区",
            "Москва, Красная площадь"
        ]
    })
    reader = DataFrameReader(df, "address")

    # Execute
    results = list(reader.read())

    # Verify
    assert len(results) == 4
    assert results[0] == "123 Main St\nApt 4"
    assert results[1] == "Café René, 456 Rue de la Paix"
    assert results[2] == "北京市朝阳区"
    assert results[3] == "Москва, Красная площадь"
