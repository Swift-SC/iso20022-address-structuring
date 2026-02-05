"""Combination generation - Single Responsibility: Generate country-town combinations."""

from data_structuring.components.database import Database
from data_structuring.components.flags import TownFlag, CountryFlag
from data_structuring.components.fuzzy_matching.fuzzy_scan import FuzzyMatch
from data_structuring.config import (
    PostProcessingConfig,
    PostProcessingTownWeightsConfig,
    PostProcessingCountryWeightsConfig
)


class CombinationGenerator:
    """Generates and scores country-town combinations."""

    def __init__(
            self,
            database: Database,
            config: PostProcessingConfig,
            town_weights: PostProcessingTownWeightsConfig,
            country_weights: PostProcessingCountryWeightsConfig
    ):
        self.database = database
        self.config = config
        self.town_weights = town_weights
        self.country_weights = country_weights
        self.suggested_country = None
        self.force_suggested_country = False

    def set_suggested_country(self, suggested_country: str | None = None):
        self.suggested_country = suggested_country

    def set_force_suggested_country(self, force_suggested_country: bool = False):
        self.force_suggested_country = force_suggested_country

    def generate_combinations(
            self,
            countries_above_threshold: list[FuzzyMatch],
            towns_above_threshold: list[FuzzyMatch],
            no_country: FuzzyMatch,
            no_town: FuzzyMatch
    ) -> list[tuple[FuzzyMatch, FuzzyMatch, float]]:
        """
        Generate scored combinations of country and town matches.

        Returns:
            List of (country, town, score) tuples, sorted by score and deduplicated
        """
        combinations = []

        # Combine matching country-town pairs
        combinations.extend(self._generate_matched_pairs(
            countries_above_threshold,
            towns_above_threshold
        ))

        # Countries without matching towns
        combinations.extend(self._generate_solo_countries(
            countries_above_threshold,
            no_town
        ))

        # Towns without matching countries
        combinations.extend(self._generate_solo_towns(
            towns_above_threshold,
            no_country
        ))

        # Default case: no country and no town
        if not combinations:
            combinations.append((no_country, no_town, (self.config.minimal_final_score_country
                                                       + self.config.minimal_final_score_town) / 2))

        # Sort by score
        combinations.sort(key=lambda x: x[2], reverse=True)

        # Remove duplicates
        return self._deduplicate_combinations(combinations)

    def _generate_matched_pairs(self,
                                countries: list[FuzzyMatch],
                                towns: list[FuzzyMatch]
                                ) -> list[tuple[FuzzyMatch, FuzzyMatch, float]]:
        """Generate combinations for matching country-town pairs."""
        combinations = []

        for town_match in towns:
            for country_match in [c for c in countries if c.origin == town_match.origin]:
                if self._should_skip_pair(town_match, country_match):
                    continue

                score = (country_match.final_score + town_match.final_score) / 2
                combinations.append((country_match, town_match, score))

        return combinations

    def _should_skip_pair(self,
                          town_match: FuzzyMatch,
                          country_match: FuzzyMatch
                          ) -> bool:
        """Check if country-town pair should be skipped."""
        # When force_suggested_country is set, filter combinations to only
        # keep those matching the suggested country code
        if self.suggested_country and self.force_suggested_country:
            if country_match.origin != self.suggested_country:
                return True

        # Same position but not a known same-name case
        if (town_match.start == country_match.start
                and town_match.end == country_match.end
                and self.database.country_town_same_name.get(town_match.possibility, "") != country_match.origin):
            return True

        # One is subset of the other
        if ((town_match.start > country_match.start and town_match.end <= country_match.end)
                or (town_match.start >= country_match.start and town_match.end < country_match.end)
                or (country_match.start > town_match.start and country_match.end <= town_match.end)
                or (country_match.start >= town_match.start and country_match.end < town_match.end)):
            return True

        return False

    def _generate_solo_countries(self,
                                 countries: list[FuzzyMatch],
                                 no_town: FuzzyMatch
                                 ) -> list[tuple[FuzzyMatch, FuzzyMatch, float]]:
        """Generate combinations for countries without matching towns."""
        combinations = []

        for country_match in countries:
            if self.suggested_country and self.force_suggested_country:
                if country_match.origin != self.suggested_country:
                    continue

            cumulative_malus = self.config.no_town_found_mul * sum([
                self.country_weights.town_is_present if (
                        CountryFlag.TOWN_IS_PRESENT in country_match.flags) else 0,
                self.country_weights.is_very_close_to_town if (
                        CountryFlag.IS_VERY_CLOSE_TO_TOWN in country_match.flags) else 0,
                self.country_weights.is_on_same_line_as_town if (
                        CountryFlag.IS_ON_SAME_LINE_AS_TOWN in country_match.flags) else 0
            ])

            score = (country_match.final_score + self.config.minimal_final_score_town - cumulative_malus) / 2
            combinations.append((country_match, no_town, score))

        return combinations

    def _generate_solo_towns(
            self,
            towns: list[FuzzyMatch],
            no_country: FuzzyMatch
    ) -> list[tuple[FuzzyMatch, FuzzyMatch, float]]:
        """Generate combinations for towns without matching countries."""
        combinations = []

        if self.suggested_country and self.force_suggested_country and self.suggested_country != "NO COUNTRY":
            return combinations

        for town_match in towns:
            cumulative_malus = self.config.no_country_found_mul * sum([
                self.town_weights.country_is_present_bonus if (
                        TownFlag.COUNTRY_IS_PRESENT in town_match.flags) else 0,
                self.town_weights.suggested_country_is_present_bonus if (
                        TownFlag.SUGGESTED_COUNTRY_IS_PRESENT in town_match.flags) else 0,
                self.town_weights.is_very_close_to_country if (
                        TownFlag.IS_VERY_CLOSE_TO_COUNTRY in town_match.flags) else 0,
                self.town_weights.is_on_same_line_as_country if (
                        TownFlag.IS_ON_SAME_LINE_AS_COUNTRY in town_match.flags) else 0
            ])

            score = (self.config.minimal_final_score_country + town_match.final_score - cumulative_malus) / 2
            combinations.append((no_country, town_match, score))

        return combinations

    def _deduplicate_combinations(
            self,
            combinations: list[tuple[FuzzyMatch, FuzzyMatch, float]]
    ) -> list[tuple[FuzzyMatch, FuzzyMatch, float]]:
        """Remove duplicate country-town combinations."""
        already_found = {}
        unique_list = []

        for combination in combinations:
            country_code = combination[0].origin
            town_name = combination[1].possibility.replace('-', ' ')

            if country_code not in already_found:
                already_found[country_code] = {town_name}
                unique_list.append(combination)
            elif town_name not in already_found[country_code]:
                already_found[country_code].add(town_name)
                unique_list.append(combination)

        return unique_list
