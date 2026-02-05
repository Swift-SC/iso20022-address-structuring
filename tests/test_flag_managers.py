from unittest.mock import MagicMock

from data_structuring.components.details import Details, TaggedSpan
from data_structuring.components.flags import TownFlag, CountryFlag, CommonFlag
from data_structuring.components.fuzzy_matching.fuzzy_scan import FuzzyMatch, FuzzyMatchResult
from data_structuring.components.runners.post_processing.flag_managers import (
    TownFlagManager, CountryFlagManager, RelationshipFlagManager
)
from data_structuring.components.runners.result_processing import ResultRunnerFuzzyMatch, ResultRunnerCRF, PredictionCRF
from data_structuring.components.tags import Tag
from data_structuring.config import PostProcessingConfig


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


def create_fuzzy_match_result(town_matches=None):
    """Helper function to create ResultRunnerFuzzyMatch instances."""
    return ResultRunnerFuzzyMatch(
        country_matches=FuzzyMatchResult([]),
        country_code_matches=FuzzyMatchResult([]),
        town_matches=FuzzyMatchResult(town_matches or []),
        extended_town_matches=FuzzyMatchResult([])
    )


def test_check_alone_on_line_town_alone():
    """Test check_alone_on_line when town is alone on its line."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data - town alone on a line
    town_match = create_fuzzy_match(
        start=16, end=21, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Address with town alone on line
    address = "1 rue de Rivoli\nParis\nFrance"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify
    assert TownFlag.IS_ALONE_ON_LINE in town_match.flags


def test_check_alone_on_line_town_not_alone():
    """Test check_alone_on_line when town is not alone on its line."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data - town with other text on line
    town_match = create_fuzzy_match(
        start=16, end=21, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Address with town not alone on line
    address = "1 rue de Rivoli Paris France"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify
    assert TownFlag.IS_ALONE_ON_LINE not in town_match.flags


def test_check_alone_on_line_town_with_spaces():
    """Test check_alone_on_line when town is alone but with surrounding spaces."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data
    town_match = create_fuzzy_match(
        start=19, end=24, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Address with spaces around town
    address = "1 rue de Rivoli\n   Paris   \nFrance"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify - spaces should be stripped, so it should be considered alone
    assert TownFlag.IS_ALONE_ON_LINE in town_match.flags


def test_check_alone_on_line_town_at_start():
    """Test check_alone_on_line when town is at the start of the address."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data - town at start
    town_match = create_fuzzy_match(
        start=0, end=5, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Address with town at start on its own line
    address = "Paris\n1 rue de Rivoli\nFrance"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify
    assert TownFlag.IS_ALONE_ON_LINE in town_match.flags


def test_check_alone_on_line_town_at_end():
    """Test check_alone_on_line when town is at the end of the address."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data - town at end
    town_match = create_fuzzy_match(
        start=23, end=28, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Address with town at end on its own line
    address = "1 rue de Rivoli\nFrance\nParis"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify
    assert TownFlag.IS_ALONE_ON_LINE in town_match.flags


def test_check_alone_on_line_single_line_address():
    """Test check_alone_on_line when entire address is on a single line."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data
    town_match = create_fuzzy_match(
        start=16, end=22, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Single line address with other text
    address = "1 rue de Rivoli Paris France"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify - not alone because there's text before and after
    assert TownFlag.IS_ALONE_ON_LINE not in town_match.flags


def test_check_alone_on_line_single_line_only_town():
    """Test check_alone_on_line when address is only the town."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data
    town_match = create_fuzzy_match(
        start=0, end=5, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Address is only the town
    address = "Paris"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify - should be alone
    assert TownFlag.IS_ALONE_ON_LINE in town_match.flags


def test_check_alone_on_line_multiple_towns():
    """Test check_alone_on_line with multiple towns."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data - two towns
    town1 = create_fuzzy_match(
        start=16, end=21, matched="Paris", possibility="PARIS", origin="FR"
    )
    town2 = create_fuzzy_match(
        start=22, end=26, matched="Lyon", possibility="LYON", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town1, town2])

    # Address with two towns on same line
    address = "1 rue de Rivoli\nParis Lyon\nFrance"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify - neither should be alone
    assert TownFlag.IS_ALONE_ON_LINE not in town1.flags
    assert TownFlag.IS_ALONE_ON_LINE not in town2.flags


def test_check_alone_on_line_multiple_towns_different_lines():
    """Test check_alone_on_line with multiple towns on different lines."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data - two towns on different lines
    town1 = create_fuzzy_match(
        start=16, end=21, matched="Paris", possibility="PARIS", origin="FR"
    )
    town2 = create_fuzzy_match(
        start=22, end=26, matched="Lyon", possibility="LYON", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town1, town2])

    # Address with two towns on different lines, each alone
    address = "1 rue de Rivoli\nParis\nLyon\nFrance"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify - both should be alone
    assert TownFlag.IS_ALONE_ON_LINE in town1.flags
    assert TownFlag.IS_ALONE_ON_LINE in town2.flags


def test_check_alone_on_line_town_with_text_before():
    """Test check_alone_on_line when town has text before it on the same line."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data
    town_match = create_fuzzy_match(
        start=22, end=27, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Address with text before town on same line
    address = "1 rue de Rivoli\nCity: Paris\nFrance"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify - not alone because of "City: " before it
    assert TownFlag.IS_ALONE_ON_LINE not in town_match.flags


def test_check_alone_on_line_town_with_text_after():
    """Test check_alone_on_line when town has text after it on the same line."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data
    town_match = create_fuzzy_match(
        start=22, end=27, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Address with text after town on same line
    address = "1 rue de Rivoli\n75001 Paris\nFrance"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify - not alone because of ", 75001" after it
    assert TownFlag.IS_ALONE_ON_LINE not in town_match.flags


def test_check_alone_on_line_empty_lines():
    """Test check_alone_on_line with empty lines around town."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data
    town_match = create_fuzzy_match(
        start=17, end=22, matched="Paris", possibility="PARIS", origin="FR"
    )
    fuzzy_result = create_fuzzy_match_result([town_match])

    # Address with empty lines
    address = "1 rue de Rivoli\n\nParis\n\nFrance"

    # Execute
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify - should be alone on its line
    assert TownFlag.IS_ALONE_ON_LINE in town_match.flags


def test_check_alone_on_line_no_towns():
    """Test check_alone_on_line with no town matches."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = TownFlagManager(mock_database, config)

    # Create test data with no towns
    fuzzy_result = create_fuzzy_match_result([])

    # Address
    address = "1 rue de Rivoli\nFrance"

    # Execute - should not raise any errors
    manager.check_alone_on_line(fuzzy_result, address)

    # Verify - no assertions needed, just checking it doesn't crash
    assert True


# ============================================================================
# CountryFlagManager Tests
# ============================================================================

def create_crf_result(country_code="US", country_code_confidence=0.95, spans=None, predictions_per_tag=None):
    """Helper function to create ResultRunnerCRF instances."""
    mock_details = MagicMock(spec=Details)
    mock_details.country_code = country_code
    mock_details.country_code_confidence = country_code_confidence
    mock_details.spans = spans or []

    return ResultRunnerCRF(
        details=mock_details,
        predictions_per_tag=predictions_per_tag or {Tag.COUNTRY: set(), Tag.TOWN: set(), Tag.POSTAL_CODE: set()},
        emissions_per_tag={},
        log_probas_per_tag={}
    )


def create_country_fuzzy_match_result(country_matches=None):
    """Helper function to create ResultRunnerFuzzyMatch with country matches."""
    return ResultRunnerFuzzyMatch(
        country_matches=FuzzyMatchResult(country_matches or []),
        country_code_matches=FuzzyMatchResult([]),
        town_matches=FuzzyMatchResult([]),
        extended_town_matches=FuzzyMatchResult([])
    )


def test_add_all_flags_separator_typo():
    """Test add_all_flags adds separator typo flag."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match with distance but separator difference
    country_match = create_fuzzy_match(
        start=0, end=13, matched="UNITED STATES", possibility="UNITED-STATES",
        origin="US", dist=1
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result()

    # Execute
    manager.add_all_flags(fuzzy_result, crf_result, "United States", "united states", [])

    # Verify
    assert CommonFlag.IS_SEPARATOR_TYPO in country_match.flags


def test_add_all_flags_iban_present():
    """Test add_all_flags adds IBAN flag when IBAN matches country."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {}
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result(country_code="FR")

    # Execute with IBAN
    manager.add_all_flags(fuzzy_result, crf_result, "France", "france", ["FR7630006000011234567890189"])

    # Verify
    assert CountryFlag.IBAN_IS_PRESENT in country_match.flags


def test_add_all_flags_no_iban():
    """Test add_all_flags does not add IBAN flag when no IBAN present."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {}
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result(country_code="FR")

    # Execute without IBAN
    manager.add_all_flags(fuzzy_result, crf_result, "France", "france", [])

    # Verify
    assert CountryFlag.IBAN_IS_PRESENT not in country_match.flags


def test_add_all_flags_province_alias_us():
    """Test add_all_flags adds province alias flag for US states."""
    # Setup
    mock_database = MagicMock()
    mock_database.provinces = {"US": ["CA", "CALIFORNIA"]}
    mock_database.countries_features = {}
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match with short US state code
    country_match = create_fuzzy_match(
        start=0, end=2, matched="CA", possibility="CA", origin="US"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result(country_code="US")

    # Execute
    manager.add_all_flags(fuzzy_result, crf_result, "CA", "ca", [])

    # Verify
    assert CommonFlag.IS_COMMON_STATE_PROVINCE_ALIAS in country_match.flags


def test_add_all_flags_province_alias_cn():
    """Test add_all_flags adds province alias flag for Chinese provinces."""
    # Setup
    mock_database = MagicMock()
    mock_database.provinces = {"CN": ["BJ", "BEIJING"]}
    mock_database.countries_features = {}
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match with short Chinese province code
    country_match = create_fuzzy_match(
        start=0, end=2, matched="BJ", possibility="BJ", origin="CN"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result(country_code="CN")

    # Execute
    manager.add_all_flags(fuzzy_result, crf_result, "BJ", "bj", [])

    # Verify
    assert CommonFlag.IS_COMMON_STATE_PROVINCE_ALIAS in country_match.flags


def test_add_all_flags_mlp_strongly_agrees():
    """Test add_all_flags adds MLP strongly agrees flag."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {}
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result(country_code="FR", country_code_confidence=0.995)

    # Execute
    manager.add_all_flags(fuzzy_result, crf_result, "France", "france", [])

    # Verify
    assert CountryFlag.MLP_STRONGLY_AGREES in country_match.flags


def test_add_all_flags_mlp_agrees():
    """Test add_all_flags adds MLP agrees flag."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {}
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result(country_code="FR", country_code_confidence=0.92)

    # Execute
    manager.add_all_flags(fuzzy_result, crf_result, "France", "france", [])

    # Verify
    assert CountryFlag.MLP_AGREES in country_match.flags
    assert CountryFlag.MLP_STRONGLY_AGREES not in country_match.flags


def test_add_all_flags_mlp_doesnt_disagree():
    """Test add_all_flags adds MLP doesn't disagree flag."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {}
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result(country_code="FR", country_code_confidence=0.60)

    # Execute
    manager.add_all_flags(fuzzy_result, crf_result, "France", "france", [])

    # Verify
    assert CountryFlag.MLP_DOESNT_DISAGREE in country_match.flags
    assert CountryFlag.MLP_AGREES not in country_match.flags


def test_add_all_flags_phone_prefix():
    """Test add_all_flags adds phone prefix flag."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {
        "FR": {"phone_prefixes": ["+33", "0033"], "domain_extensions": [], "postal_code_regex": None}
    }
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result(country_code="FR")

    # Execute with phone prefix in sample
    manager.add_all_flags(fuzzy_result, crf_result, "France +33 1 23 45 67 89", "france +33 1 23 45 67 89", [])

    # Verify
    assert CountryFlag.PHONE_PREFIX_IS_PRESENT in country_match.flags


def test_add_all_flags_domain_extension():
    """Test add_all_flags adds domain extension flag."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {
        "FR": {"phone_prefixes": [], "domain_extensions": [".fr"], "postal_code_regex": None}
    }
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])
    crf_result = create_crf_result(country_code="FR")

    # Execute with domain in sample
    manager.add_all_flags(fuzzy_result, crf_result, "France contact@example.fr", "france contact@example.fr", [])

    # Verify
    assert CountryFlag.DOMAIN_IS_PRESENT in country_match.flags


def test_add_all_flags_postal_code():
    """Test add_all_flags adds postal code flag."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {
        "FR": {"phone_prefixes": [], "domain_extensions": [], "postal_code_regex": r"^\d{5}$"}
    }
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create country match
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])

    # Create CRF result with postal code prediction
    postal_prediction = PredictionCRF(
        tag=Tag.POSTAL_CODE,
        start=7,
        end=12,
        confidence=0.9,
        prediction="75001"
    )
    crf_result = create_crf_result(
        country_code="FR",
        predictions_per_tag={Tag.POSTAL_CODE: {postal_prediction}, Tag.COUNTRY: set(), Tag.TOWN: set()}
    )

    # Execute
    manager.add_all_flags(fuzzy_result, crf_result, "France 75001", "france 75001", [])

    # Verify
    assert CountryFlag.POSTAL_CODE_IS_PRESENT in country_match.flags


def test_add_all_flags_street_intersection():
    """Test add_all_flags adds street intersection flag."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {}
    config = PostProcessingConfig()
    config.part_of_street_ratio = 0.5
    manager = CountryFlagManager(mock_database, config)

    # Create country match that overlaps with street
    country_match = create_fuzzy_match(
        start=0, end=10, matched="Washington", possibility="WASHINGTON", origin="US"
    )
    fuzzy_result = create_country_fuzzy_match_result([country_match])

    # Create CRF result with street span overlapping the country match
    street_span = TaggedSpan(tag=Tag.STREET, start=0, end=15)
    crf_result = create_crf_result(country_code="US", spans=[street_span])

    # Execute
    manager.add_all_flags(fuzzy_result, crf_result, "Washington St.", "washington st.", [])

    # Verify
    assert CommonFlag.IS_INSIDE_STREET in country_match.flags


def test_add_all_flags_multiple_countries():
    """Test add_all_flags processes multiple country matches."""
    # Setup
    mock_database = MagicMock()
    mock_database.countries_features = {}
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create multiple country matches
    country1 = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR"
    )
    country2 = create_fuzzy_match(
        start=10, end=17, matched="Germany", possibility="GERMANY", origin="DE"
    )
    fuzzy_result = create_country_fuzzy_match_result([country1, country2])
    crf_result = create_crf_result(country_code="FR", country_code_confidence=0.995)

    # Execute
    manager.add_all_flags(fuzzy_result, crf_result, "France and Germany", "france and germany", [])

    # Verify - only France should have MLP flag since CRF predicts FR
    assert CountryFlag.MLP_STRONGLY_AGREES in country1.flags
    assert CountryFlag.MLP_STRONGLY_AGREES not in country2.flags


def test_add_all_flags_no_countries():
    """Test add_all_flags with no country matches."""
    # Setup
    mock_database = MagicMock()
    config = PostProcessingConfig()
    manager = CountryFlagManager(mock_database, config)

    # Create empty fuzzy result
    fuzzy_result = create_country_fuzzy_match_result([])
    crf_result = create_crf_result()

    # Execute - should not raise any errors
    manager.add_all_flags(fuzzy_result, crf_result, "Some address", "some address", [])

    # Verify - no assertions needed, just checking it doesn't crash
    assert True


# ============================================================================
# RelationshipFlagManager Tests
# ============================================================================

def create_full_fuzzy_match_result(country_matches=None, town_matches=None):
    """Helper function to create ResultRunnerFuzzyMatch with both country and town matches."""
    return ResultRunnerFuzzyMatch(
        country_matches=FuzzyMatchResult(country_matches or []),
        country_code_matches=FuzzyMatchResult([]),
        town_matches=FuzzyMatchResult(town_matches or []),
        extended_town_matches=FuzzyMatchResult([])
    )


def test_add_relationship_flags_matching_origin():
    """Test add_relationship_flags with matching country and town origins."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create matching country and town
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    town_match = create_fuzzy_match(
        start=11, end=16, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute
    manager.add_relationship_flags(fuzzy_result, "France and Paris", None)

    # Verify - both should have presence flags
    assert TownFlag.COUNTRY_IS_PRESENT in town_match.flags
    assert CountryFlag.TOWN_IS_PRESENT in country_match.flags


def test_add_relationship_flags_different_origins():
    """Test add_relationship_flags with different country and town origins."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create non-matching country and town
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    town_match = create_fuzzy_match(
        start=11, end=17, matched="Berlin", possibility="BERLIN", origin="DE", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute
    manager.add_relationship_flags(fuzzy_result, "France and Berlin", None)

    # Verify - no flags should be added
    assert TownFlag.COUNTRY_IS_PRESENT not in town_match.flags
    assert CountryFlag.TOWN_IS_PRESENT not in country_match.flags


def test_add_relationship_flags_with_distance():
    """Test add_relationship_flags skips matches with distance > 0."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create country and town with distance
    country_match = create_fuzzy_match(
        start=0, end=6, matched="Frnace", possibility="FRANCE", origin="FR", dist=1
    )
    town_match = create_fuzzy_match(
        start=11, end=16, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute
    manager.add_relationship_flags(fuzzy_result, "Frnace and Paris", None)

    # Verify - no flags should be added due to distance
    assert TownFlag.COUNTRY_IS_PRESENT not in town_match.flags
    assert CountryFlag.TOWN_IS_PRESENT not in country_match.flags


def test_add_relationship_flags_close_proximity():
    """Test add_relationship_flags adds proximity flags when close."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create close country and town (within 15 chars)
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    town_match = create_fuzzy_match(
        start=8, end=13, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute - only 2 chars between (", ")
    manager.add_relationship_flags(fuzzy_result, "France, Paris", None)

    # Verify - should have proximity flags
    assert TownFlag.IS_VERY_CLOSE_TO_COUNTRY in town_match.flags
    assert CountryFlag.IS_VERY_CLOSE_TO_TOWN in country_match.flags


def test_add_relationship_flags_far_apart():
    """Test add_relationship_flags does not add proximity flags when far apart."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create far apart country and town (> 15 chars)
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    town_match = create_fuzzy_match(
        start=25, end=30, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute - more than 15 chars between
    manager.add_relationship_flags(fuzzy_result, "France                   Paris", None)

    # Verify - should have presence flags but not proximity flags
    assert TownFlag.COUNTRY_IS_PRESENT in town_match.flags
    assert TownFlag.IS_VERY_CLOSE_TO_COUNTRY not in town_match.flags


def test_add_relationship_flags_same_line():
    """Test add_relationship_flags adds same line flags."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create country and town on same line
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    town_match = create_fuzzy_match(
        start=8, end=13, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute - no newline between
    manager.add_relationship_flags(fuzzy_result, "France, Paris", None)

    # Verify - should have same line flags
    assert TownFlag.IS_ON_SAME_LINE_AS_COUNTRY in town_match.flags
    assert CountryFlag.IS_ON_SAME_LINE_AS_TOWN in country_match.flags


def test_add_relationship_flags_different_lines():
    """Test add_relationship_flags does not add same line flags when on different lines."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create country and town on different lines
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    town_match = create_fuzzy_match(
        start=7, end=12, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute - newline between
    manager.add_relationship_flags(fuzzy_result, "France\nParis", None)

    # Verify - should have presence flags but not same line flags
    assert TownFlag.COUNTRY_IS_PRESENT in town_match.flags
    assert TownFlag.IS_ON_SAME_LINE_AS_COUNTRY not in town_match.flags


def test_add_relationship_flags_extended_data():
    """Test add_relationship_flags with extended town data."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create country and town with extended data flag
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    town_match = create_fuzzy_match(
        start=8, end=13, matched="Paris", possibility="PARIS", origin="FR", dist=0,
        flags=[TownFlag.IS_FROM_EXTENDED_DATA]
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute
    manager.add_relationship_flags(fuzzy_result, "France, Paris", None)

    # Verify - town should have flags but country should not (extended data)
    assert TownFlag.COUNTRY_IS_PRESENT in town_match.flags
    assert CountryFlag.TOWN_IS_PRESENT not in country_match.flags
    assert TownFlag.IS_VERY_CLOSE_TO_COUNTRY in town_match.flags
    assert CountryFlag.IS_VERY_CLOSE_TO_TOWN not in country_match.flags


def test_add_relationship_flags_province_alias():
    """Test add_relationship_flags skips additional flags for province aliases."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create country with province alias flag
    country_match = create_fuzzy_match(
        start=0, end=2, matched="CA", possibility="CA", origin="US", dist=0,
        flags=[CommonFlag.IS_COMMON_STATE_PROVINCE_ALIAS]
    )
    town_match = create_fuzzy_match(
        start=4, end=14, matched="Sacramento", possibility="SACRAMENTO", origin="US", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute
    manager.add_relationship_flags(fuzzy_result, "CA, Sacramento", None)

    # Verify - should have presence flags but not proximity flags (ambiguous)
    assert TownFlag.COUNTRY_IS_PRESENT in town_match.flags
    assert TownFlag.IS_VERY_CLOSE_TO_COUNTRY not in town_match.flags


def test_add_relationship_flags_mlp_country_present():
    """Test add_relationship_flags adds MLP country present flag."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create town match
    town_match = create_fuzzy_match(
        start=0, end=5, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([], [town_match])

    # Execute with matching country_head
    manager.add_relationship_flags(fuzzy_result, "Paris", "FR")

    # Verify
    assert TownFlag.MLP_COUNTRY_IS_PRESENT in town_match.flags


def test_add_relationship_flags_mlp_country_not_present():
    """Test add_relationship_flags does not add MLP flag when country_head doesn't match."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create town match
    town_match = create_fuzzy_match(
        start=0, end=5, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([], [town_match])

    # Execute with non-matching country_head
    manager.add_relationship_flags(fuzzy_result, "Paris", "DE")

    # Verify
    assert TownFlag.MLP_COUNTRY_IS_PRESENT not in town_match.flags


def test_add_relationship_flags_town_inside_word():
    """Test add_relationship_flags skips town with IS_INSIDE_ANOTHER_WORD flag."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create country and town with town inside word
    country_match = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    town_match = create_fuzzy_match(
        start=11, end=16, matched="Paris", possibility="PARIS", origin="FR", dist=0,
        flags=[CommonFlag.IS_INSIDE_ANOTHER_WORD]
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute
    manager.add_relationship_flags(fuzzy_result, "France and Paris", None)

    # Verify - no flags should be added
    assert TownFlag.COUNTRY_IS_PRESENT not in town_match.flags


def test_add_relationship_flags_country_short_inside_word():
    """Test add_relationship_flags skips country with both SHORT and INSIDE_WORD flags."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create country with problematic flags
    country_match = create_fuzzy_match(
        start=0, end=2, matched="US", possibility="US", origin="US", dist=0,
        flags=[CommonFlag.IS_SHORT, CommonFlag.IS_INSIDE_ANOTHER_WORD]
    )
    town_match = create_fuzzy_match(
        start=7, end=14, matched="Seattle", possibility="SEATTLE", origin="US", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute
    manager.add_relationship_flags(fuzzy_result, "US and Seattle", None)

    # Verify - no flags should be added
    assert TownFlag.COUNTRY_IS_PRESENT not in town_match.flags


def test_add_relationship_flags_multiple_pairs():
    """Test add_relationship_flags with multiple country-town pairs."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create multiple countries and towns
    country1 = create_fuzzy_match(
        start=0, end=6, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    country2 = create_fuzzy_match(
        start=18, end=25, matched="Germany", possibility="GERMANY", origin="DE", dist=0
    )
    town1 = create_fuzzy_match(
        start=8, end=13, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    town2 = create_fuzzy_match(
        start=27, end=33, matched="Berlin", possibility="BERLIN", origin="DE", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country1, country2], [town1, town2])

    # Execute
    manager.add_relationship_flags(fuzzy_result, "France, Paris and Germany, Berlin", None)

    # Verify - each matching pair should have flags
    assert TownFlag.COUNTRY_IS_PRESENT in town1.flags
    assert CountryFlag.TOWN_IS_PRESENT in country1.flags
    assert TownFlag.COUNTRY_IS_PRESENT in town2.flags
    assert CountryFlag.TOWN_IS_PRESENT in country2.flags


def test_add_relationship_flags_no_matches():
    """Test add_relationship_flags with no matches."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create empty fuzzy result
    fuzzy_result = create_full_fuzzy_match_result([], [])

    # Execute - should not raise any errors
    manager.add_relationship_flags(fuzzy_result, "Some address", None)

    # Verify - no assertions needed, just checking it doesn't crash
    assert True


def test_add_relationship_flags_town_before_country():
    """Test add_relationship_flags when town appears before country."""
    # Setup
    mock_database = MagicMock()
    manager = RelationshipFlagManager(mock_database)

    # Create town before country
    country_match = create_fuzzy_match(
        start=8, end=14, matched="France", possibility="FRANCE", origin="FR", dist=0
    )
    town_match = create_fuzzy_match(
        start=0, end=5, matched="Paris", possibility="PARIS", origin="FR", dist=0
    )
    fuzzy_result = create_full_fuzzy_match_result([country_match], [town_match])

    # Execute
    manager.add_relationship_flags(fuzzy_result, "Paris, France", None)

    # Verify - should still work correctly
    assert TownFlag.COUNTRY_IS_PRESENT in town_match.flags
    assert CountryFlag.TOWN_IS_PRESENT in country_match.flags
    assert TownFlag.IS_VERY_CLOSE_TO_COUNTRY in town_match.flags
    assert TownFlag.IS_ON_SAME_LINE_AS_COUNTRY in town_match.flags
