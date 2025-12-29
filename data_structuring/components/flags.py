"""
Module providing the flag classes.
"""
from enum import StrEnum


class BaseFlag(StrEnum):
    pass


class TownFlag(BaseFlag):
    # The country containing the town is in the message
    COUNTRY_IS_PRESENT = "COUNTRY_IS_PRESENT"
    # The MLP infers the country containing the town is in the message
    MLP_COUNTRY_IS_PRESENT = "MLP_COUNTRY_IS_PRESENT"
    # We are very close to the country of that town in the message
    IS_VERY_CLOSE_TO_COUNTRY = "IS_VERY_CLOSE_TO_COUNTRY"
    # ... if we are even on the same line
    IS_ON_SAME_LINE_AS_COUNTRY = "IS_ON_SAME_LINE_AS_COUNTRY"
    # Population above 500K
    IS_METROPOLIS = "IS_METROPOLIS"
    # population below 12K
    IS_SMALL_TOWN = "IS_SMALL_TOWN"
    # The match comes from the extended dataset
    IS_FROM_EXTENDED_DATA = "IS_FROM_EXTENDED_DATA"
    # The town is alone on its line
    IS_ALONE_ON_LINE = "IS_ALONE_ON_LINE"
    # The town origin match is the largest of all possibilities for a town with this name
    IS_NOT_LARGEST_TOWN_WITH_NAME = "IS_NOT_LARGEST_TOWN_WITH_NAME"
    # The postcode for the town and country combo was found via the postcode matcher
    POSTCODE_FOR_TOWN_FOUND = "POSTCODE_FOR_TOWN_FOUND"


class CommonFlag(BaseFlag):
    # Typo in town name comes from missing/additional separators
    IS_SEPARATOR_TYPO = "IS_SEPARATOR_TYPO"
    # The match was found inside another word (for example, 'IL' is inside 'MILITARY'
    IS_INSIDE_ANOTHER_WORD = "IS_INSIDE_ANOTHER_WORD"
    # The town or country is detected in the first third part of the message
    IS_IN_FIRST_THIRD = "IS_IN_FIRST_THIRD"
    # The town or country is detected in the last third part of the message
    IS_IN_LAST_THIRD = "IS_IN_LAST_THIRD"
    # The town or country detected has 2 or less characters
    IS_SHORT = "IS_SHORT"
    # After all matches have been computed, there is another match that includes us
    # (example: 'IL' and 'CHILE' are both country matches)
    # Two possibilities depending on whether said match is higher or lower ranked than us
    IS_INSIDE_ANOTHER_HIGHER_RANKED_MATCH = "IS_INSIDE_ANOTHER_HIGHER_RANKED_MATCH"
    IS_INSIDE_ANOTHER_LOWER_RANKED_MATCH = "IS_INSIDE_ANOTHER_LOWER_RANKED_MATCH"
    # If the match is a town, did the CRF think it could have been a country ? Or vice-versa ?
    COULD_BE_REASONABLE_MISTAKE = "COULD_BE_REASONABLE_MISTAKE"
    # The match is inside a span that was tagged as a street by the CRF
    IS_INSIDE_STREET = "IS_INSIDE_STREET"
    # If the match is on an alias of a province that can be confused with a country ISO code
    IS_COMMON_STATE_PROVINCE_ALIAS = "IS_COMMON_STATE_PROVINCE_ALIAS"
    # The match is on an alias of an Indian province which can be confused with a country ISO code or other abbreviation
    IS_UNCOMMON_STATE_PROVINCE_ALIAS = "IS_UNCOMMON_STATE_PROVINCE_ALIAS"


class CountryFlag(BaseFlag):
    # There is at least one town in that country in the message
    TOWN_IS_PRESENT = "TOWN_IS_PRESENT"
    # We are very close to at least one town of that country in the message
    IS_VERY_CLOSE_TO_TOWN = "IS_VERY_CLOSE_TO_TOWN"
    # If we are even on the same line
    IS_ON_SAME_LINE_AS_TOWN = "IS_ON_SAME_LINE_AS_TOWN"
    # There is at least one postal code **identified by the crf** that is in that country in the message
    POSTAL_CODE_IS_PRESENT = "POSTAL_CODE_IS_PRESENT"
    # There is at least one IBAN of that country in the message
    IBAN_IS_PRESENT = "IBAN_IS_PRESENT"
    # There is at least one match for a phone prefix (e.g., "+32") related to that country in the message
    PHONE_PREFIX_IS_PRESENT = "PHONE_PREFIX_IS_PRESENT"
    # There is at least one match for a internet domain suffix (e.g., ".be") related to that country in the message
    DOMAIN_IS_PRESENT = "DOMAIN_IS_PRESENT"
    # The MLP country prediction result is between 99% to 100%
    MLP_STRONGLY_AGREES = "MLP_STRONGLY_AGREES"
    # The MLP country prediction result is between 90% to 98%
    MLP_AGREES = "MLP_AGREES"
    # The MLP country prediction result is between 50% to 89%
    MLP_DOESNT_DISAGREE = "MLP_DOESNT_DISAGREE"
