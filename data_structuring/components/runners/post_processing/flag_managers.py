"""
Flag managers - Single Responsibility: Each manager handles specific flag types.
"""
import re
from typing import Optional

from data_structuring.components.data_provider import decode_and_clean_str
from data_structuring.components.database import Database
from data_structuring.components.flags import TownFlag, CountryFlag, CommonFlag
from data_structuring.components.fuzzy_matching.fuzzy_scan import FuzzyMatch, FuzzyMatchResult
from data_structuring.components.runners.result_processing import ResultRunnerCRF, ResultRunnerFuzzyMatch
from data_structuring.components.tags import Tag
from data_structuring.config import PostProcessingConfig


class BaseFlagManager:
    """Base class with shared flagging methods to avoid code duplication."""

    def __init__(self, database: Database, config: PostProcessingConfig):
        self.database = database
        self.config = config

    @staticmethod
    def _normalize_string(text: str) -> str:
        """Remove separators for comparison (optimized)."""
        return text.replace('-', '').replace(' ', '')

    @staticmethod
    def _add_separator_typo_flag(match: FuzzyMatch) -> None:
        """Check if typo is due to separator differences."""
        if match.dist > 0:
            if BaseFlagManager._normalize_string(match.matched) == BaseFlagManager._normalize_string(match.possibility):
                match.flags.append(CommonFlag.IS_SEPARATOR_TYPO)

    def _add_street_intersection_flag(self, match: FuzzyMatch, crf_result: ResultRunnerCRF) -> None:
        """Check if match overlaps significantly with street spans (optimized)."""
        match_start, match_end = match.start, match.end
        match_length = match_end - match_start
        n_chars_in_street = 0

        for span in crf_result.details.spans:
            if span.start > match_end:
                break  # Spans are likely sorted, early exit
            if span.tag == Tag.STREET:
                # Calculate overlap without creating sets (memory efficient)
                overlap_start = max(span.start, match_start)
                overlap_end = min(span.end, match_end)
                if overlap_start < overlap_end:
                    n_chars_in_street += overlap_end - overlap_start

        if n_chars_in_street > 0:
            ratio_part_of_street = n_chars_in_street / match_length
            if ratio_part_of_street >= self.config.part_of_street_ratio:
                match.flags.append(CommonFlag.IS_INSIDE_STREET)


class MatchInclusionFlagger:
    """Flags matches that are included within other matches."""

    @staticmethod
    def flag_matches_included_in_another(
            queries: list[FuzzyMatch],
            larger_matches: FuzzyMatchResult
    ) -> None:
        """
        Flag matches included in other matches with position-aware flags.

        Args:
            queries: Matches to check for inclusion
            larger_matches: Potentially larger matches to check against
        """
        for i, match in enumerate(queries):
            for j, other_match in enumerate(larger_matches):
                # Check if strictly larger
                left_larger = (other_match.start < match.start) and (other_match.end >= match.end)
                right_larger = (other_match.end > match.end) and (other_match.start <= match.start)

                if left_larger or right_larger:
                    if i <= j and other_match.dist < 1:
                        match.flags.append(CommonFlag.IS_INSIDE_ANOTHER_LOWER_RANKED_MATCH)
                    elif i > j:
                        match.flags.append(CommonFlag.IS_INSIDE_ANOTHER_HIGHER_RANKED_MATCH)


class TownFlagManager(BaseFlagManager):
    """Manages town-specific flags."""

    def add_all_flags(self, fuzzy_match_result: ResultRunnerFuzzyMatch, crf_result: ResultRunnerCRF) -> None:
        """Add all town-specific flags to town matches."""
        for town_match in fuzzy_match_result.town_matches:
            self._add_separator_typo_flag(town_match)
            self._add_population_flags(town_match)
            self._add_street_intersection_flag(town_match, crf_result)

    def _add_population_flags(self, match: FuzzyMatch) -> None:
        """Add population-based flags (metropolis, small town, etc.)."""
        town_name = decode_and_clean_str(match.possibility.casefold())
        town_pop = self.database.towns_and_populations.get(town_name)

        if town_pop is not None:
            if town_pop >= self.config.is_metropolis_threshold:
                match.flags.append(TownFlag.IS_METROPOLIS)
            if town_pop <= self.config.is_small_town_threshold:
                match.flags.append(TownFlag.IS_SMALL_TOWN)
            if match.origin != self.database.largest_country_code_for_town.get(town_name):
                match.flags.append(TownFlag.IS_NOT_LARGEST_TOWN_WITH_NAME)

    def check_alone_on_line(self, fuzzy_match_result: ResultRunnerFuzzyMatch, original_address: str) -> None:
        """Check if towns are alone on their line (optimized)."""
        for town_match in fuzzy_match_result.town_matches:
            before = original_address[:town_match.start]
            after = original_address[town_match.end:]

            newline_before = before.rfind("\n")
            before_line = before[newline_before + 1:] if newline_before != -1 else before

            newline_after = after.find("\n")
            after_line = after[:newline_after] if newline_after != -1 else after

            # Optimized: strip() is faster than all(c == ' ')
            is_start_of_line = before_line.strip() == ''
            is_end_of_line = after_line.strip() == ''

            if is_start_of_line and is_end_of_line:
                town_match.flags.append(TownFlag.IS_ALONE_ON_LINE)


class CountryFlagManager(BaseFlagManager):
    """Manages country-specific flags."""

    def add_all_flags(self,
                      fuzzy_match_result: ResultRunnerFuzzyMatch,
                      crf_result: ResultRunnerCRF,
                      sample: str,
                      sample_casefold: str,
                      ibans: list[str]
                      ) -> None:
        """Add all country-specific flags to country matches."""
        for country_match in fuzzy_match_result.country_matches:
            self._add_separator_typo_flag(country_match)
            self._add_iban_flag(country_match, ibans)
            self._add_street_intersection_flag(country_match, crf_result)
            self._add_province_alias_flags(country_match)
            self._add_crf_agreement_flags(country_match, crf_result)
            self._add_feature_flags(country_match, crf_result, sample, sample_casefold)

    def _add_iban_flag(self, match: FuzzyMatch, ibans: list[str]) -> None:
        """Check if IBAN country code matches (optimized)."""
        if match.origin and ibans:
            origin_lower = match.origin.casefold()
            for iban in ibans:
                if len(iban) >= 2 and iban[:2].casefold() == origin_lower:
                    match.flags.append(CountryFlag.IBAN_IS_PRESENT)
                    break

    def _add_province_alias_flags(self, match: FuzzyMatch) -> None:
        """Flag problematic province/state aliases."""
        if len(match.possibility) <= 2 and match.origin and match.origin in self.database.provinces:
            flag = (CommonFlag.IS_COMMON_STATE_PROVINCE_ALIAS
                    if match.origin in self.config.countries_with_common_provinces
                    else CommonFlag.IS_UNCOMMON_STATE_PROVINCE_ALIAS)
            aliases_list = self.database.provinces.get(match.origin, [])
            if match.possibility in aliases_list:
                match.flags.append(flag)

    def _add_crf_agreement_flags(self, match: FuzzyMatch, crf_result: ResultRunnerCRF) -> None:
        """Add flags based on CRF country prediction confidence."""
        if match.origin and match.origin == crf_result.details.country_code:
            country_head_score = crf_result.details.country_code_confidence * 100
            if country_head_score >= 99:
                match.flags.append(CountryFlag.MLP_STRONGLY_AGREES)
            elif country_head_score >= 90:
                match.flags.append(CountryFlag.MLP_AGREES)
            elif country_head_score >= 50:
                match.flags.append(CountryFlag.MLP_DOESNT_DISAGREE)

    def _add_feature_flags(self,
                           match: FuzzyMatch,
                           crf_result: ResultRunnerCRF,
                           sample: str,
                           sample_casefold: str
                           ) -> None:
        """Add flags for country-specific features (phone, domain, postal code)."""
        if not match.origin:
            return

        country_features = self.database.countries_features.get(match.origin)
        if not country_features:
            return

        # Phone prefix check
        phone_prefixes = country_features.get("phone_prefixes", [])
        if phone_prefixes and any(phone_prefix in sample for phone_prefix in phone_prefixes):
            match.flags.append(CountryFlag.PHONE_PREFIX_IS_PRESENT)

        # Domain extensions check
        domain_extensions = country_features.get("domain_extensions", [])
        if domain_extensions and any(domain_ext in sample_casefold for domain_ext in domain_extensions):
            match.flags.append(CountryFlag.DOMAIN_IS_PRESENT)

        # Postal codes check
        postal_regex = country_features.get("postal_code_regex")
        if postal_regex:
            postal_predictions = crf_result.predictions_per_tag.get(Tag.POSTAL_CODE, [])
            if postal_predictions:
                # Compile regex once for reuse
                pattern = re.compile(postal_regex)
                if any(pattern.search(pred.prediction) for pred in postal_predictions):
                    match.flags.append(CountryFlag.POSTAL_CODE_IS_PRESENT)


class RelationshipFlagManager:
    """Manages flags for country-town relationships."""

    def __init__(self, database: Database):
        self.database = database

    def add_relationship_flags(self,
                               fuzzy_match_result: ResultRunnerFuzzyMatch,
                               original_address: str,
                               country_head: Optional[str]
                               ) -> None:
        """Add flags indicating country-town relationships."""
        for town_match in fuzzy_match_result.town_matches:
            for country_match in fuzzy_match_result.country_matches:
                self._check_pair_validity(town_match, country_match, original_address)

            if country_head == town_match.origin:
                town_match.flags.append(TownFlag.MLP_COUNTRY_IS_PRESENT)

    def _check_pair_validity(self,
                             town_match: FuzzyMatch,
                             country_match: FuzzyMatch,
                             original_address: str
                             ) -> None:
        """Check if country-town pair is valid and add appropriate flags (optimized)."""
        # Skip if not reliable matches
        if (country_match.dist > 0 or town_match.dist > 0
                or CommonFlag.IS_INSIDE_ANOTHER_WORD in town_match.flags):
            return

        # Skip if country has problematic flags
        if (CommonFlag.IS_SHORT in country_match.flags
                and CommonFlag.IS_INSIDE_ANOTHER_WORD in country_match.flags):
            return

        # Skip if origins don't match
        if not country_match.origin or country_match.origin != town_match.origin:
            return

        # Add basic presence flags
        is_extended_data = TownFlag.IS_FROM_EXTENDED_DATA in town_match.flags
        if (CountryFlag.IS_SUGGESTED_COUNTRY in country_match.flags
                or CountryFlag.GENERATED_BY_SUGGESTED_COUNTRY in country_match.flags):
            town_match.flags.append(TownFlag.SUGGESTED_COUNTRY_IS_PRESENT)
        town_match.flags.append(TownFlag.COUNTRY_IS_PRESENT)
        if not is_extended_data:
            country_match.flags.append(CountryFlag.TOWN_IS_PRESENT)

        # Skip ambiguous cases for additional flags
        if (CommonFlag.IS_COMMON_STATE_PROVINCE_ALIAS in country_match.flags
                or CommonFlag.IS_UNCOMMON_STATE_PROVINCE_ALIAS in country_match.flags):
            return

        # Calculate distance between matches
        if town_match.start <= country_match.start:
            string_between = original_address[town_match.end:country_match.start]
        else:
            string_between = original_address[country_match.end:town_match.start]

        character_distance = len(string_between)

        # Add proximity flags except if pair contains a synthetically generated match
        if character_distance <= 15 and CountryFlag.GENERATED_BY_SUGGESTED_COUNTRY not in country_match.flags:
            town_match.flags.append(TownFlag.IS_VERY_CLOSE_TO_COUNTRY)
            if not is_extended_data:
                country_match.flags.append(CountryFlag.IS_VERY_CLOSE_TO_TOWN)

        if "\n" not in string_between and CountryFlag.GENERATED_BY_SUGGESTED_COUNTRY not in country_match.flags:
            town_match.flags.append(TownFlag.IS_ON_SAME_LINE_AS_COUNTRY)
            if not is_extended_data:
                country_match.flags.append(CountryFlag.IS_ON_SAME_LINE_AS_TOWN)

    def check_reasonable_mistakes(self,
                                  fuzzy_match_result: ResultRunnerFuzzyMatch,
                                  crf_result: ResultRunnerCRF
                                  ) -> None:
        """Flag matches that could have been confused by CRF."""
        for town_match in fuzzy_match_result.town_matches:
            if any(town_match.matched in pred.prediction
                   for pred in crf_result.predictions_per_tag[Tag.COUNTRY]):
                town_match.flags.append(CommonFlag.COULD_BE_REASONABLE_MISTAKE)

        for country_match in fuzzy_match_result.country_matches:
            if any(country_match.matched in pred.prediction
                   for pred in crf_result.predictions_per_tag[Tag.TOWN]):
                country_match.flags.append(CommonFlag.COULD_BE_REASONABLE_MISTAKE)
