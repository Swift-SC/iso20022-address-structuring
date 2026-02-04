from data_structuring.components.data_provider.country_provider import get_country_overrides
from data_structuring.config import DatabaseConfig


def test_get_country_overrides_with_groupings(tmp_path):
    """Test get_country_overrides with country groupings."""
    # Create a temporary country groupings file
    test_data = tmp_path / "test_country_groupings.json"
    test_data.write_text('''
    {
        "EU": ["FR", "DE", "IT"],
        "AMERICAS": ["US", "CA", "MX"],
        "ASIA": ["SG", "MY", "TH"]
    }
    ''')

    # Create a mock config
    config = DatabaseConfig(
        country_groupings=test_data,
        force_country_groupings=["EU", "AMERICAS"],
        force_countries=["JP", "KR"]
    )

    # Call the function
    result = get_country_overrides(config)

    # Verify the result
    expected = {"FR", "DE", "IT", "US", "CA", "MX", "JP", "KR"}
    assert len(result) == len(expected)
    assert set(result) == expected


def test_get_country_overrides_no_groupings():
    """Test get_country_overrides with no country groupings."""
    # Create a mock config with no groupings
    config = DatabaseConfig(
        force_country_groupings=[],
        force_countries=["JP", "KR", "CN"]
    )

    # Call the function
    result = get_country_overrides(config)

    # Verify only force_countries are returned
    assert set(result) == {"JP", "KR", "CN"}


def test_get_country_overrides_empty_grouping(tmp_path):
    """Test get_country_overrides with an empty country grouping."""
    # Create a temporary country groupings file with empty group
    test_data = tmp_path / "empty_groupings.json"
    test_data.write_text('{"EU": []}')

    # Create a mock config
    config = DatabaseConfig(
        country_groupings=test_data,
        force_country_groupings=["EU"],
        force_countries=["US"]
    )

    # Call the function
    result = get_country_overrides(config)

    # Verify only force_countries are returned
    assert result == ["US"]


def test_get_country_overrides_nonexistent_grouping(tmp_path):
    """Test get_country_overrides with a non-existent country grouping."""
    # Create a temporary country groupings file
    test_data = tmp_path / "test_groupings.json"
    test_data.write_text('{"EU": ["FR", "DE"]}')

    # Create a mock config with a non-existent group
    config = DatabaseConfig(
        country_groupings=test_data,
        force_country_groupings=["NON_EXISTENT"],
        force_countries=["US"]
    )

    # Call the function
    result = get_country_overrides(config)

    # Verify only force_countries are returned
    assert result == ["US"]
