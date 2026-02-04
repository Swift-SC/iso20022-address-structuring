import pytest

from data_structuring.components.data_provider.normalization import (
    duplicate_if_saint_in_name,
    duplicate_if_separator_present,
    decode_and_clean_str
)


@pytest.mark.parametrize(
    "input_name, expected_output",
    [
        (
                "SAINT-ETIENNE",
                {"SAINT-ETIENNE", "ST. ETIENNE", "ST-ETIENNE"},
        ),
        (
                "ST. JOHN'S",
                {"SAINT-JOHN'S", "ST. JOHN'S", "ST-JOHN'S"},
        ),
        (
                "ST JULIANS",
                {"SAINT-JULIANS", "ST. JULIANS", "ST-JULIANS"},
        ),
        (
                "SAINT PETERSBURG",
                {"SAINT-PETERSBURG", "ST. PETERSBURG", "ST-PETERSBURG"},
        ),
        (
                "AINT-ETIENNE",
                {"AINT-ETIENNE"},
        )
    ],
)
def test_duplicate_if_saint_in_name(input_name, expected_output):
    assert duplicate_if_saint_in_name(input_name) == expected_output


@pytest.mark.parametrize(
    "input_name, expected_output",
    [
        ("Val-d'Oise", {"Val-d'Oise", "Val d'Oise"}),
        ("Val d'Oise", {"Val-d'Oise", "Val d'Oise"}),
        ("NoSeparator", {"NoSeparator"}),
        (
                "Multiple-separators-present",
                {"Multiple-separators-present", "Multiple separators present"},
        ),
        (
                "Multiple separators present",
                {"Multiple-separators-present", "Multiple separators present"},
        )
    ]
)
def test_duplicate_if_separator_present(input_name, expected_output):
    assert duplicate_if_separator_present(input_name) == expected_output


@pytest.mark.parametrize(
    "input_name, expected_output",
    [
        ("this needs to be replaced: `test @ Rock´n roll` and ‘–others–‘",
         "this needs to be replaced: 'test a Rock'n roll' and '-others-'"),
    ]
)
def test_unidecode_and_clean_char_replace(input_name, expected_output):
    assert decode_and_clean_str(input_name) == expected_output
