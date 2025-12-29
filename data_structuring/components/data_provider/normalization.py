"""
Module providing helpers functions to normalize town names and possibly create some variants
"""
import re

import polars as pl
from anyascii import anyascii

# alias for the Unicode to ascii transliteration function
DECODE_ALIAS = anyascii
# Most common special characters that occur after a run of anyascii
CHAR_REPLACEMENT = {
    "@": "a",
    "`": "'",
    "–": "-"
}

SEP_VARIATIONS = [
    '-',
    ' '
]

SAINT_VARIATIONS_REGEX = re.compile(r'\bSAINT([^\s\w]'
                                    r'|\s)|\bST\.([^\s\w]|\s)'
                                    r'|\bST([^\s\w]|\s)')

SAINT_VARIATIONS = [
    'SAINT-',
    'ST. ',
    'ST-'
]


def decode_and_clean_str(string: str) -> str:
    """
    Transliterate a string to ascii and some replace special characters
    Args:
        string: input unicode string
    Returns:
        an ascii string
    """
    result = DECODE_ALIAS(string)
    for char, replacement in CHAR_REPLACEMENT.items():
        result = result.replace(char, replacement)
    return result


# polars-only version to use the more optimal API
def decode_and_clean_expr(expr: pl.Expr) -> pl.Expr:
    """
    Transliterate a polars Expression from unicode to ascii (and some replace special characters)
    Args:
        expr: input unicode polars Expression
    Returns:
        an ascii polars Expression
    """
    return expr.map_elements(DECODE_ALIAS, return_dtype=pl.String()).str.replace_many(CHAR_REPLACEMENT)


def duplicate_if_separator_present(name):
    processed_name = name
    for variation in SEP_VARIATIONS:
        processed_name = processed_name.replace(variation, '%TOKEN%')
    return set(processed_name.replace('%TOKEN%', variation) for variation in SEP_VARIATIONS)


def duplicate_if_saint_in_name(name):
    processed_name = SAINT_VARIATIONS_REGEX.sub('%TOKEN%', name)
    return set(processed_name.replace('%TOKEN%', variation) for variation in SAINT_VARIATIONS)


def generate_duplicate_aliases(name):
    # we keep the original entry first in the set of aliases
    return ([name] + [name_alias for variation in duplicate_if_saint_in_name(name)
                      for name_alias in duplicate_if_separator_present(variation)])
