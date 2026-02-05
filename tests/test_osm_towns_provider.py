"""Unit tests for extended towns provider module."""
import tempfile
from pathlib import Path

import polars as pl
import pytest

from data_structuring.components.data_provider.osm_towns_provider import get_extended_towns
from data_structuring.config import DatabaseConfig


@pytest.fixture
def sample_osm_data():
    """Create sample OSM data for testing."""
    return pl.DataFrame({
        'city_name': ['Paris', 'Paris', 'London', 'Berlin', 'Springfield', 'New York', 'Saint-Tropez'],
        'original_city_name': ['Paris', 'Paris', 'London', 'Berlin', 'Springfield', 'New York', 'St. Tropez'],
        'iso': ['FR', 'US', 'GB', 'DE', 'US', 'US', 'FR'],
        'population': [2_165_000, 25_000, 8_982_000, 3_645_000, 5_000, 8_336_000, 4_400],
        'label': ['name', 'name', 'name', 'name', 'name', 'name', 'name'],
        'place_type': ['city', 'town', 'city', 'city', 'town', 'city', 'town']
    })


@pytest.fixture
def temp_osm_parquet(sample_osm_data):
    """Create a temporary parquet file with sample OSM data."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        sample_osm_data.write_parquet(tmp_file.name)
        yield Path(tmp_file.name)
        # Cleanup
        Path(tmp_file.name).unlink(missing_ok=True)


@pytest.fixture
def test_config(temp_osm_parquet):
    """Create a test database configuration."""
    config = DatabaseConfig()
    config.town_entities_osm = temp_osm_parquet
    config.town_minimal_population = 10_000
    config.enable_osm_data = True
    return config


class TestLoadFromOSM:
    """Tests for _load_from_osm function."""

    def test_basic_functionality(self, test_config):
        """Test that the function loads and processes OSM data correctly."""
        towns_to_remove = {}
        country_override = []

        all_possibilities, populations, largest_countries = get_extended_towns(
            test_config, towns_to_remove, country_override
        )

        # Basic assertions
        assert isinstance(all_possibilities, dict)
        assert isinstance(populations, dict)
        assert isinstance(largest_countries, dict)

        # Check
        assert len(populations) == 5
        assert len(largest_countries) == 5

    def test_population_filtering(self, test_config):
        """Test that towns below minimal population are filtered out."""
        towns_to_remove = {}
        country_override = []

        _, populations, _ = get_extended_towns(
            test_config, towns_to_remove, country_override
        )

        # Springfield (5,000) and St. Tropez (4,400) should be filtered out since min is 10,000
        # and it's not in country_override
        assert 'springfield' not in populations
        assert 'saint tropez' not in populations
        assert 'saint-tropez' not in populations

        # Paris, London, Berlin, New York should be included
        assert 'paris' in populations
        assert 'london' in populations
        assert 'berlin' in populations
        assert 'new york' in populations

    def test_country_override(self, test_config):
        """Test that country_override includes towns below minimal population."""
        towns_to_remove = {}
        country_override = ['US']

        _, populations, _ = get_extended_towns(
            test_config, towns_to_remove, country_override
        )

        # Springfield (US, 5,000) should now be included due to country override but not St. Tropez
        assert 'springfield' in populations
        assert 'saint tropez' not in populations

        # check with multiple countries
        country_override = ['US', 'FR']

        _, populations, _ = get_extended_towns(test_config, towns_to_remove, country_override)

        assert 'springfield' in populations
        assert 'saint tropez' in populations
        assert 'saint-tropez' in populations

    def test_highest_population_selection(self, test_config):
        """Test that the highest population is selected for duplicate towns."""
        towns_to_remove = {}
        country_override = []

        _, populations, largest_countries = get_extended_towns(
            test_config, towns_to_remove, country_override
        )

        # Paris exists in both FR (2,165,000) and US (25,000)
        # Should keep the FR version with higher population
        assert populations['paris'] == 2_165_000
        assert largest_countries['paris'] == 'FR'

    def test_towns_to_remove(self, test_config):
        """Test that specified towns are removed from results."""
        towns_to_remove = {'london', 'berlin'}
        country_override = []

        _, populations, _ = get_extended_towns(
            test_config, towns_to_remove, country_override
        )

        # London and Berlin should be removed
        assert 'london' not in populations
        assert 'berlin' not in populations

        # Paris should still be present
        assert 'paris' in populations

    def test_iso_list_mapping(self, test_config):
        """Test that ISO codes are correctly mapped to towns."""
        towns_to_remove = {}
        country_override = []

        all_possibilities, _, _ = get_extended_towns(
            test_config, towns_to_remove, country_override
        )

        # Check that ISO codes are properly mapped
        assert 'FR' in all_possibilities
        assert 'GB' in all_possibilities
        assert 'US' in all_possibilities

        # Paris should be in both FR and US lists
        assert 'paris' in all_possibilities['FR']
        assert 'paris' in all_possibilities['US']


class TestDataIntegrity:
    """Tests for data integrity and consistency."""

    def test_set_update_efficiency(self, test_config):
        """Test that ISO sets are properly updated for all towns."""
        towns_to_remove = {}
        country_override = []

        all_possibilities, _, _ = get_extended_towns(
            test_config, towns_to_remove, country_override
        )

        # For Paris (appears in both FR and US)
        if 'paris' in all_possibilities.get('FR', {}):
            paris_fr_isos = all_possibilities['FR']['paris']
            # Should contain both FR and US ISO codes
            assert 'FR' in paris_fr_isos
            assert 'US' in paris_fr_isos

    def test_defaultdict_initialization(self, test_config):
        """Test that defaultdicts are properly initialized."""
        towns_to_remove = {}
        country_override = []

        all_possibilities, populations, largest_countries = get_extended_towns(
            test_config, towns_to_remove, country_override
        )

        # Accessing a non-existent key should not raise an error
        # (defaultdict behavior)
        test_iso = all_possibilities['NONEXISTENT']
        assert isinstance(test_iso, dict)

        # populations defaultdict should return 0 for non-existent keys
        assert populations['nonexistent_town'] == 0
