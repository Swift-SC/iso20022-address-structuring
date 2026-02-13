import pytest

from data_structuring.components.flags import TownFlag, CountryFlag, CommonFlag
from data_structuring.components.runners.post_processing.score_computer import ScoreComputer
from data_structuring.config import PostProcessingTownWeightsConfig, PostProcessingCountryWeightsConfig


@pytest.fixture
def score_computer():
    """Create a ScoreComputer instance with default configs."""
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()
    return ScoreComputer(town_weights, country_weights)


# ============================================================================
# Town Score Tests
# ============================================================================

def test_compute_town_score_basic(score_computer):
    """Test compute_town_score with basic inputs."""
    # High CRF score, no distance, no flags
    score = score_computer.compute_town_score(crf_score=0.9, dist_score=0, flags=[])

    # Should return a valid probability
    assert 0 <= score <= 1
    assert score > 0.5  # High CRF should give high score


def test_compute_town_score_extreme_crf_values(score_computer):
    """Test compute_town_score with extreme CRF values."""
    # Very low CRF score
    low_score = score_computer.compute_town_score(crf_score=0.01, dist_score=0, flags=[])
    assert 0 <= low_score <= 1
    assert low_score < 0.5

    # Very high CRF score
    high_score = score_computer.compute_town_score(crf_score=0.99, dist_score=0, flags=[])
    assert 0 <= high_score <= 1
    assert high_score > 0.5


def test_compute_town_score_with_distance(score_computer):
    """Test compute_town_score with fuzzy match distance."""
    # Score without distance
    score_no_dist = score_computer.compute_town_score(crf_score=0.8, dist_score=0, flags=[])

    # Score with distance
    score_with_dist = score_computer.compute_town_score(crf_score=0.8, dist_score=2, flags=[])

    # Distance should decrease score
    assert score_with_dist < score_no_dist


def test_compute_town_score_separator_typo_exception(score_computer):
    """Test that IS_SEPARATOR_TYPO flag prevents distance penalty."""
    # Score with distance but separator typo flag
    score_separator = score_computer.compute_town_score(
        crf_score=0.8, dist_score=2, flags=[CommonFlag.IS_SEPARATOR_TYPO]
    )

    # Score with no distance
    score_no_dist = score_computer.compute_town_score(crf_score=0.8, dist_score=0, flags=[])

    # Should be similar since separator typo negates distance penalty
    assert abs(score_separator - score_no_dist) < 0.1


def test_compute_town_score_with_bonus_flags(score_computer):
    """Test compute_town_score with bonus flags."""
    # Score without flags
    score_no_flags = score_computer.compute_town_score(crf_score=0.7, dist_score=0, flags=[])

    # Score with bonus flags
    score_with_bonus = score_computer.compute_town_score(
        crf_score=0.7, dist_score=0,
        flags=[TownFlag.COUNTRY_IS_PRESENT, TownFlag.IS_VERY_CLOSE_TO_COUNTRY]
    )

    # Bonus flags should increase score
    assert score_with_bonus > score_no_flags


def test_compute_town_score_bonus(score_computer):
    for flag in [
        TownFlag.IS_METROPOLIS,
        TownFlag.POSTCODE_FOR_TOWN_FOUND,
        TownFlag.MLP_COUNTRY_IS_PRESENT,
        TownFlag.IS_ALONE_ON_LINE,
    ]:
        score_no_flag = score_computer.compute_town_score(crf_score=0.7, dist_score=0, flags=[])
        score_with_flag = score_computer.compute_town_score(crf_score=0.7, dist_score=0, flags=[flag])

        assert score_with_flag > score_no_flag


def test_compute_town_score_with_malus_flags(score_computer):
    """Test compute_town_score with multiple malus flags."""
    # Score without flags
    score_no_flags = score_computer.compute_town_score(crf_score=0.7, dist_score=0, flags=[])

    # Score with malus flags
    score_with_malus = score_computer.compute_town_score(
        crf_score=0.7, dist_score=0,
        flags=[CommonFlag.IS_INSIDE_ANOTHER_WORD, CommonFlag.IS_SHORT]
    )

    # Malus flags should decrease score
    assert score_with_malus < score_no_flags


def test_compute_town_score_malus(score_computer):
    for flag in [
        TownFlag.IS_SMALL_TOWN,
        TownFlag.IS_FROM_EXTENDED_DATA,
        CommonFlag.IS_INSIDE_ANOTHER_WORD,
        CommonFlag.IS_SHORT,
    ]:
        score_no_flag = score_computer.compute_town_score(crf_score=0.7, dist_score=0, flags=[])
        score_with_flag = score_computer.compute_town_score(crf_score=0.7, dist_score=0, flags=[flag]
                                                            )

    assert score_with_flag < score_no_flag


def test_compute_town_score_small_town_no_country_malus(score_computer):
    """Test compute_town_score with IS_SMALL_TOWN without COUNTRY_IS_PRESENT."""
    # Small town with country present
    score_with_country = score_computer.compute_town_score(
        crf_score=0.7, dist_score=0,
        flags=[TownFlag.IS_SMALL_TOWN, TownFlag.COUNTRY_IS_PRESENT]
    )

    # Small town without country present (double penalty)
    score_no_country = score_computer.compute_town_score(
        crf_score=0.7, dist_score=0,
        flags=[TownFlag.IS_SMALL_TOWN]
    )

    # Should be lower without country
    assert score_no_country < score_with_country


def test_compute_town_score_short_with_distance(score_computer):
    """Test compute_town_score with IS_SHORT and distance (double penalty)."""
    # Short with no distance
    score_short_no_dist = score_computer.compute_town_score(
        crf_score=0.7, dist_score=0, flags=[CommonFlag.IS_SHORT]
    )

    # Short with distance (additional penalty)
    score_short_with_dist = score_computer.compute_town_score(
        crf_score=0.7, dist_score=2, flags=[CommonFlag.IS_SHORT]
    )

    # Should be lower with distance
    assert score_short_with_dist < score_short_no_dist


def test_compute_town_score_combined_flags(score_computer):
    """Test compute_town_score with multiple bonus and malus flags."""
    score = score_computer.compute_town_score(
        crf_score=0.7, dist_score=1,
        flags=[
            TownFlag.COUNTRY_IS_PRESENT,  # Bonus
            TownFlag.IS_METROPOLIS,  # Bonus
            CommonFlag.IS_SHORT,  # Malus
            CommonFlag.IS_IN_LAST_THIRD  # Bonus
        ]
    )

    # Should still be a valid probability
    assert 0 <= score <= 1


def test_compute_town_score_returns_valid_probability(score_computer):
    """Test that compute_town_score always returns valid probabilities."""
    test_cases = [
        (0.1, 0, []),
        (0.5, 0, []),
        (0.9, 0, []),
        (0.7, 3, []),
        (0.7, 0, [TownFlag.IS_METROPOLIS, TownFlag.COUNTRY_IS_PRESENT]),
        (0.7, 0, [CommonFlag.IS_SHORT, CommonFlag.IS_INSIDE_ANOTHER_WORD]),
        (0.01, 5, [CommonFlag.IS_SHORT]),
        (0.99, 0, [TownFlag.IS_METROPOLIS, TownFlag.IS_ALONE_ON_LINE]),
    ]

    for crf_score, dist_score, flags in test_cases:
        score = score_computer.compute_town_score(crf_score, dist_score, flags)
        assert 0 <= score <= 1, f"Invalid score {score} for inputs: {crf_score}, {dist_score}, {flags}"


# ============================================================================
# Country Score Tests
# ============================================================================

def test_compute_country_score_basic(score_computer):
    """Test compute_country_score with basic inputs."""
    # High CRF score, no distance, no flags
    score = score_computer.compute_country_score(crf_score=0.9, dist_score=0, flags=[])

    # Should return a valid probability
    assert 0 <= score <= 1
    assert score > 0.5  # High CRF should give high score


def test_compute_country_score_extreme_crf_values(score_computer):
    """Test compute_country_score with extreme CRF values."""
    # Very low CRF score
    low_score = score_computer.compute_country_score(crf_score=0.01, dist_score=0, flags=[])
    assert 0 <= low_score <= 1
    assert low_score < 0.5

    # Very high CRF score
    high_score = score_computer.compute_country_score(crf_score=0.99, dist_score=0, flags=[])
    assert 0 <= high_score <= 1
    assert high_score > 0.5


def test_compute_country_score_with_distance(score_computer):
    """Test compute_country_score with fuzzy match distance."""
    # Score without distance
    score_no_dist = score_computer.compute_country_score(crf_score=0.8, dist_score=0, flags=[])

    # Score with distance
    score_with_dist = score_computer.compute_country_score(crf_score=0.8, dist_score=2, flags=[])

    # Distance should decrease score
    assert score_with_dist < score_no_dist


def test_compute_country_score_separator_typo_exception(score_computer):
    """Test that IS_SEPARATOR_TYPO flag prevents distance penalty."""
    # Score with distance but separator typo flag
    score_separator = score_computer.compute_country_score(
        crf_score=0.8, dist_score=2, flags=[CommonFlag.IS_SEPARATOR_TYPO]
    )

    # Score with no distance
    score_no_dist = score_computer.compute_country_score(crf_score=0.8, dist_score=0, flags=[])

    # Should be similar since separator typo negates distance penalty
    assert abs(score_separator - score_no_dist) < 0.1


def test_compute_country_score_with_multiple_bonus(score_computer):
    """Test compute_country_score with bonus flags."""
    # Score without flags
    score_no_flags = score_computer.compute_country_score(crf_score=0.7, dist_score=0, flags=[])

    # Score with bonus flags
    score_with_bonus = score_computer.compute_country_score(
        crf_score=0.7, dist_score=0,
        flags=[CountryFlag.TOWN_IS_PRESENT, CountryFlag.IBAN_IS_PRESENT]
    )

    # Bonus flags should increase score
    assert score_with_bonus > score_no_flags


def test_compute_country_score_bonus(score_computer):
    score_no_flags = score_computer.compute_country_score(crf_score=0.7, dist_score=0, flags=[])
    for flag in [
        CountryFlag.MLP_STRONGLY_AGREES,
        CountryFlag.MLP_AGREES,
        CountryFlag.MLP_DOESNT_DISAGREE,
        CountryFlag.IBAN_IS_PRESENT,
        CountryFlag.PHONE_PREFIX_IS_PRESENT,
        CountryFlag.DOMAIN_IS_PRESENT,
        CountryFlag.POSTAL_CODE_IS_PRESENT]:
        # Score with bonus flags
        score_with_bonus = score_computer.compute_country_score(
            crf_score=0.7, dist_score=0,
            flags=[flag]
        )

        # Bonus flags should increase score
        assert score_with_bonus > score_no_flags


def test_compute_country_score_malus(score_computer):
    score_no_flags = score_computer.compute_country_score(crf_score=0.7, dist_score=0, flags=[])
    for flag in [
        CommonFlag.IS_INSIDE_ANOTHER_WORD,
        CommonFlag.IS_SHORT,
        CommonFlag.IS_COMMON_STATE_PROVINCE_ALIAS,
        CommonFlag.IS_INSIDE_STREET]:
        # Score with malus flags
        score_with_malus = score_computer.compute_country_score(
            crf_score=0.7, dist_score=0,
            flags=[flag]
        )

        # Malus flags should decrease score
        assert score_with_malus < score_no_flags


def test_compute_country_score_with_multiple_malus(score_computer):
    """Test compute_country_score with multiple malus flags."""
    # Score without flags
    score_no_flags = score_computer.compute_country_score(crf_score=0.7, dist_score=0, flags=[])

    # Score with malus flags
    score_with_malus = score_computer.compute_country_score(
        crf_score=0.7, dist_score=0,
        flags=[CommonFlag.IS_INSIDE_ANOTHER_WORD, CommonFlag.IS_SHORT]
    )

    # Malus flags should decrease score
    assert score_with_malus < score_no_flags


def test_compute_country_score_combined_flags(score_computer):
    """Test compute_country_score with multiple bonus and malus flags."""
    score = score_computer.compute_country_score(
        crf_score=0.7, dist_score=1,
        flags=[
            CountryFlag.TOWN_IS_PRESENT,  # Bonus
            CountryFlag.IBAN_IS_PRESENT,  # Bonus
            CommonFlag.IS_SHORT,  # Malus
            CommonFlag.IS_IN_LAST_THIRD  # Bonus
        ]
    )

    # Should still be a valid probability
    assert 0 <= score <= 1


def test_compute_country_score_combined_flags_and_was_suggested_country(score_computer):
    """Test compute_country_score with multiple bonus and malus flags."""
    score = score_computer.compute_country_score(
        crf_score=0.7, dist_score=1,
        flags=[
            CountryFlag.IS_SUGGESTED_COUNTRY,  # No effect; used for explainability
            CountryFlag.TOWN_IS_PRESENT,  # Bonus
            CountryFlag.IBAN_IS_PRESENT,  # Bonus
            CommonFlag.IS_SHORT,  # Malus
            CommonFlag.IS_IN_LAST_THIRD  # Bonus
        ]
    )

    # Should still be a valid probability
    assert 0 <= score <= 1


def test_compute_country_score_when_generated_by_suggested_country(score_computer):
    """Test compute_country_score when the GENERATED_BY_SUGGESTED_COUNTRY flag is present."""
    score = score_computer.compute_country_score(
        crf_score=0.7, dist_score=0,
        flags=[
            CountryFlag.GENERATED_BY_SUGGESTED_COUNTRY
        ]
    )

    # Should be the same as the crf_score
    assert score == 0.7


def test_compute_country_score_returns_valid_probability(score_computer):
    """Test that compute_country_score always returns valid probabilities."""
    test_cases = [
        (0.1, 0, []),
        (0.5, 0, []),
        (0.9, 0, []),
        (0.7, 3, []),
        (0.7, 0, [CountryFlag.IBAN_IS_PRESENT, CountryFlag.TOWN_IS_PRESENT]),
        (0.7, 0, [CommonFlag.IS_SHORT, CommonFlag.IS_INSIDE_ANOTHER_WORD]),
        (0.01, 5, [CommonFlag.IS_SHORT]),
        (0.99, 0, [CountryFlag.MLP_STRONGLY_AGREES, CountryFlag.POSTAL_CODE_IS_PRESENT]),
    ]

    for crf_score, dist_score, flags in test_cases:
        score = score_computer.compute_country_score(crf_score, dist_score, flags)
        assert 0 <= score <= 1, f"Invalid score {score} for inputs: {crf_score}, {dist_score}, {flags}"


# ============================================================================
# Edge Cases and Comparison Tests
# ============================================================================

def test_score_computer_initialization():
    """Test ScoreComputer initialization with custom configs."""
    town_weights = PostProcessingTownWeightsConfig()
    country_weights = PostProcessingCountryWeightsConfig()

    computer = ScoreComputer(town_weights, country_weights)

    assert computer.town_weights == town_weights
    assert computer.country_weights == country_weights


def test_town_vs_country_score_consistency(score_computer):
    """Test that town and country scores behave consistently."""
    # Both should produce valid probabilities for same inputs
    town_score = score_computer.compute_town_score(0.7, 0, [])
    country_score = score_computer.compute_country_score(0.7, 0, [])

    assert 0 <= town_score <= 1
    assert 0 <= country_score <= 1


def test_score_monotonicity_with_crf(score_computer):
    """Test that higher CRF scores generally lead to higher final scores."""
    scores = []
    for crf in [0.1, 0.3, 0.5, 0.7, 0.9]:
        score = score_computer.compute_town_score(crf, 0, [])
        scores.append(score)

    # Scores should generally increase with CRF
    for i in range(len(scores) - 1):
        assert scores[i] < scores[i + 1]
