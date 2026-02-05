from unittest.mock import MagicMock

from data_structuring.components.flags import CountryFlag
from data_structuring.components.fuzzy_matching.fuzzy_scan import FuzzyMatch
from data_structuring.components.runners.post_processing.combination_generator import CombinationGenerator
from data_structuring.config import (
    PostProcessingConfig,
    PostProcessingTownWeightsConfig,
    PostProcessingCountryWeightsConfig
)


def create_fuzzy_match(
        start=0,
        end=5,
        matched="test",
        possibility="test",
        dist=0,
        origin="US",
        final_score=0.9,
        flags=None
):
    """Helper function to create FuzzyMatch instances for testing."""
    return FuzzyMatch(
        start=start,
        end=end,
        matched=matched,
        possibility=possibility,
        dist=dist,
        origin=origin,
        final_score=final_score,
        flags=flags or []
    )


def test_generate_combinations_with_matched_pairs():
    """Test generate_combinations with matching country-town pairs."""
    # Setup
    mock_database = MagicMock()
    mock_database.country_town_same_name = {}
    config = PostProcessingConfig()
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()

    generator = CombinationGenerator(mock_database, config, town_weights, country_weights)

    # Create test data - matching country and town from same origin
    country1 = create_fuzzy_match(
        start=0, end=2, matched="North America", possibility="UNITED STATES",
        origin="US", final_score=0.9
    )
    country2 = create_fuzzy_match(
        start=0, end=2, matched="North America", possibility="CANADA",
        origin="CA", final_score=0.95
    )

    town1 = create_fuzzy_match(
        start=10, end=17, matched="Seattle", possibility="SEATTLE",
        origin="US", final_score=0.85
    )
    town2 = create_fuzzy_match(
        start=10, end=17, matched="New York", possibility="NEW-YORK",
        origin="US", final_score=0.75
    )
    no_country = create_fuzzy_match(possibility="NO COUNTRY", origin="NO COUNTRY", final_score=0.0)
    no_town = create_fuzzy_match(possibility="NO TOWN", origin="NO TOWN", final_score=0.0)

    # Execute
    result = generator.generate_combinations([country1, country2],
                                             [town1, town2],
                                             no_country,
                                             no_town)

    # Verify
    assert len(result) > 0
    assert result[0][0].origin == "US"  # Country
    assert result[0][1].possibility == "SEATTLE"  # Town
    assert result[0][2] == (0.9 + 0.85) / 2  # Score

    assert result[1][0].origin == "US"  # Country
    assert result[1][1].possibility == "NEW-YORK"  # Town
    assert result[1][2] == (0.9 + 0.75) / 2  # Score


def test_generate_combinations_solo_country():
    """Test generate_combinations with country but no matching town."""
    # Setup
    mock_database = MagicMock()
    mock_database.country_town_same_name = {}
    config = PostProcessingConfig()
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()

    generator = CombinationGenerator(mock_database, config, town_weights, country_weights)

    # Create test data - country without matching town
    country = create_fuzzy_match(
        start=0, end=2, matched="US", possibility="UNITED STATES",
        origin="US", final_score=0.9
    )
    no_country = create_fuzzy_match(possibility="NO COUNTRY", origin="NO COUNTRY", final_score=0.0)
    no_town = create_fuzzy_match(possibility="NO TOWN", origin="NO TOWN", final_score=0.0)

    # Execute
    result = generator.generate_combinations([country], [], no_country, no_town)

    # Verify
    assert len(result) > 0
    assert result[0][0].origin == "US"  # Country
    assert result[0][1].possibility == "NO TOWN"  # No town


def test_generate_combinations_solo_town():
    """Test generate_combinations with town but no matching country."""
    # Setup
    mock_database = MagicMock()
    mock_database.country_town_same_name = {}
    config = PostProcessingConfig()
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()

    generator = CombinationGenerator(mock_database, config, town_weights, country_weights)

    # Create test data - town without matching country
    town = create_fuzzy_match(
        start=10, end=17, matched="Seattle", possibility="SEATTLE",
        origin="US", final_score=0.85
    )
    no_country = create_fuzzy_match(possibility="NO COUNTRY", origin="NO COUNTRY", final_score=0.0)
    no_town = create_fuzzy_match(possibility="NO TOWN", origin="NO TOWN", final_score=0.0)

    # Execute
    result = generator.generate_combinations([], [town], no_country, no_town)

    # Verify
    assert len(result) > 0
    assert result[0][0].possibility == "NO COUNTRY"  # No country
    assert result[0][1].possibility == "SEATTLE"  # Town


def test_generate_combinations_no_matches():
    """Test generate_combinations with no country or town matches."""
    # Setup
    mock_database = MagicMock()
    mock_database.country_town_same_name = {}
    config = PostProcessingConfig()
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()

    generator = CombinationGenerator(mock_database, config, town_weights, country_weights)

    # Create test data - no matches
    no_country = create_fuzzy_match(possibility="NO COUNTRY", origin="NO COUNTRY", final_score=0.0)
    no_town = create_fuzzy_match(possibility="NO TOWN", origin="NO TOWN", final_score=0.0)

    # Execute
    result = generator.generate_combinations([], [], no_country, no_town)

    # Verify - should return default combination
    assert len(result) == 1
    assert result[0][0].possibility == "NO COUNTRY"
    assert result[0][1].possibility == "NO TOWN"


def test_generate_combinations_sorted_by_score():
    """Test that combinations are sorted by score in descending order."""
    # Setup
    mock_database = MagicMock()
    mock_database.country_town_same_name = {}
    config = PostProcessingConfig()
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()

    generator = CombinationGenerator(mock_database, config, town_weights, country_weights)

    # Create test data with different scores
    country1 = create_fuzzy_match(
        start=0, end=2, matched="US", possibility="UNITED STATES",
        origin="US", final_score=0.9
    )
    country2 = create_fuzzy_match(
        start=20, end=22, matched="FR", possibility="FRANCE",
        origin="FR", final_score=0.7
    )
    town1 = create_fuzzy_match(
        start=10, end=17, matched="Seattle", possibility="SEATTLE",
        origin="US", final_score=0.85
    )
    town2 = create_fuzzy_match(
        start=30, end=35, matched="Paris", possibility="PARIS",
        origin="FR", final_score=0.95
    )
    no_country = create_fuzzy_match(possibility="NO COUNTRY", origin="NO COUNTRY", final_score=0.0)
    no_town = create_fuzzy_match(possibility="NO TOWN", origin="NO TOWN", final_score=0.0)

    # Execute
    result = generator.generate_combinations([country1, country2], [town1, town2], no_country, no_town)

    # Verify - results should be sorted by score descending
    assert len(result) >= 2
    for i in range(len(result) - 1):
        assert result[i][2] >= result[i + 1][2]


def test_generate_combinations_deduplication():
    """Test that duplicate combinations are removed."""
    # Setup
    mock_database = MagicMock()
    mock_database.country_town_same_name = {}
    config = PostProcessingConfig()
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()

    generator = CombinationGenerator(mock_database, config, town_weights, country_weights)

    # Create test data with potential duplicates
    country = create_fuzzy_match(
        start=0, end=2, matched="US", possibility="UNITED STATES",
        origin="US", final_score=0.9
    )
    town1 = create_fuzzy_match(
        start=10, end=17, matched="Seattle", possibility="SEATTLE",
        origin="US", final_score=0.85
    )
    town2 = create_fuzzy_match(
        start=30, end=37, matched="Seattle", possibility="SEATTLE",
        origin="US", final_score=0.80
    )
    no_country = create_fuzzy_match(possibility="NO COUNTRY", origin="NO COUNTRY", final_score=0.0)
    no_town = create_fuzzy_match(possibility="NO TOWN", origin="NO TOWN", final_score=0.0)

    # Execute
    result = generator.generate_combinations([country], [town1, town2], no_country, no_town)

    # Verify - should only have one US-SEATTLE combination (the higher scored one)
    us_seattle_count = sum(1 for r in result if r[0].origin == "US" and r[1].possibility == "SEATTLE")
    assert us_seattle_count == 1


def test_generate_combinations_skips_overlapping_positions():
    """Test that combinations with overlapping positions are skipped."""
    # Setup
    mock_database = MagicMock()
    mock_database.country_town_same_name = {}
    config = PostProcessingConfig()
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()

    generator = CombinationGenerator(mock_database, config, town_weights, country_weights)

    # Create test data with overlapping positions
    country = create_fuzzy_match(
        start=0, end=10, matched="New York", possibility="NEW YORK",
        origin="US", final_score=0.9
    )
    town = create_fuzzy_match(
        start=0, end=10, matched="New York", possibility="NEW YORK",
        origin="US", final_score=0.85
    )
    no_country = create_fuzzy_match(possibility="NO COUNTRY", origin="NO COUNTRY", final_score=0.0)
    no_town = create_fuzzy_match(possibility="NO TOWN", origin="NO TOWN", final_score=0.0)

    # Execute
    result = generator.generate_combinations([country], [town], no_country, no_town)

    # Verify - should skip the overlapping pair and generate solo combinations
    matched_pairs = [r for r in result if r[0].origin != "NO COUNTRY" and r[1].possibility != "NO TOWN"]
    assert len(matched_pairs) == 0  # No matched pairs due to overlap


def test_generate_combinations_with_flags():
    """Test generate_combinations with flags affecting scores."""
    # Setup
    mock_database = MagicMock()
    mock_database.country_town_same_name = {}
    config = PostProcessingConfig()
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()

    generator = CombinationGenerator(mock_database, config, town_weights, country_weights)

    # Create test data with flags
    country = create_fuzzy_match(
        start=0, end=2, matched="US", possibility="UNITED STATES",
        origin="US", final_score=0.9,
        flags=[CountryFlag.TOWN_IS_PRESENT]
    )
    no_country = create_fuzzy_match(possibility="NO COUNTRY", origin="NO COUNTRY", final_score=0.0)
    no_town = create_fuzzy_match(possibility="NO TOWN", origin="NO TOWN", final_score=0.0)

    # Execute
    result = generator.generate_combinations([country], [], no_country, no_town)

    # Verify - flags should affect the score
    assert len(result) > 0
    # Score should be adjusted based on flags
    assert result[0][2] < (0.9 + config.minimal_final_score_town) / 2
