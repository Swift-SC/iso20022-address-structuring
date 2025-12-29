"""
This module contains classes for fuzzy string matching and fuzzy match results.

Fuzzy string matching is a technique used to find similar strings. It allows for a certain amount of
typos or errors in the input string. The most common technique is the Levenshtein distance, which
computes the minimum number of operations (insertions, deletions and substitutions) to transform
one string into another.

FuzzyMatchResult is a class that represents a single match. It contains the start and end indices of
the matched string within the original string, the matched string itself, the possibility (the string
that was searched for), the distance between the two strings and a list of flags that were raised
for this match.

The module also contains a class FuzzyMatch that makes use of the `fuzzysearch` and `rapidfuzz`
libraries to perform fuzzy string matching.
"""
import re

import numpy as np
from fuzzysearch import find_near_matches
from pydantic import BaseModel, RootModel, NonNegativeInt, Field, FiniteFloat
from rapidfuzz.fuzz import partial_ratio
from rapidfuzz.process import cdist

from data_structuring.components.flags import BaseFlag, CommonFlag


class FuzzyMatch(BaseModel):
    """
    Represents a single result of the fuzzy match
    """
    # Fuzzymatch-related attributes
    start: NonNegativeInt = Field(description="The absolute index of the first letter of the match within the message")
    end: NonNegativeInt = Field(description="The absolute index of the last letter of the match within the message")
    matched: str = Field(description="The string that triggered the match within the message")
    possibility: str = Field(description="The actual string that was looked for")
    dist: NonNegativeInt = Field(description="The distance between `matched` and `possibility`")
    flags: list[BaseFlag] = Field(default_factory=list, description="The list of all the flags "
                                                                    "that were raised for this match")
    origin: str | None = Field(default=None, description="The country from which `possibility` originates.")
    # CRF-related attributes
    crf_score: FiniteFloat | None = Field(default=None, description="The CRF score of this match")
    transformer_score: FiniteFloat | None = Field(default=None, description="The transformer score of this match")
    # Final score attribute
    final_score: FiniteFloat | None = Field(default=None,
                                            description="The final score of this match,"
                                                        " computed a-posteriori based on the other fields.")


class FuzzyMatchResult(RootModel[list[FuzzyMatch]]):
    """
    This class is a `RootModel` that contains a list of `FuzzyMatch` objects. It is used to represent
    a list of fuzzy matches for a given text.
    """
    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, idx):
        return self.root[idx]

    def __len__(self):
        return len(self.root)

    def sort(self, *args, **kwargs):
        return self.root.sort(*args, **kwargs)

    @classmethod
    def merge(cls, r1: "FuzzyMatchResult", r2: "FuzzyMatchResult") -> "FuzzyMatchResult":
        """
        Merge two `FuzzyMatchResult` objects into a new `FuzzyMatchResult` object.

        Args:
            r1 (FuzzyMatchResult): The first `FuzzyMatchResult` object to merge.
            r2 (FuzzyMatchResult): The second `FuzzyMatchResult` object to merge.

        Returns:
            A new `FuzzyMatchResult` object that contains the concatenated root lists of `r1` and `r2`.
        """
        return cls(r1.root + r2.root)


def fuzzyscan_all_batched(
        queries: list[str],
        words_mapping: dict[str, str],
        n_workers: int = 4,
        score_cutoff: float = 50,
        max_l_dist: float = 1
) -> list[FuzzyMatchResult]:
    """
    Perform an efficient, batched, fuzzy_scan between `queries` and the keys of `words_mapping`.
    Matches that have a partial ratio score above `score_cutoff`, or a distance above `max_l_dist` are discarded.
    """

    batched_fuzzy_match_results = []

    match_against = [match.upper() for match in words_mapping.keys()]
    names = list(words_mapping.values())

    # Filter potential candidates and only keep the ones with a score above `score_cutoff`.
    matrix_results = cdist(
        queries, match_against,
        scorer=partial_ratio,
        score_cutoff=score_cutoff,
        workers=n_workers
    )

    # Loop over all possible candidates
    for input_text, matrix_result in zip(queries, matrix_results):
        matches = []
        candidates = np.flatnonzero(matrix_result)
        for candidate in candidates:

            # Look for occurence(s) of the candidate string
            candidate_str = match_against[candidate]
            # Short aliases or country codes should match exactly
            max_dist = 0 if len(candidate_str) <= 2 else max_l_dist
            all_matches = find_near_matches(candidate_str, input_text, max_l_dist=max_dist)

            if len(all_matches) == 0:
                continue

            for match in all_matches:

                flags: list[CommonFlag] = []

                try:
                    # If the match ends with a '.' it can bug the regular expression boundary logic
                    query_match = match.matched[:-1] if match.matched.endswith('.') else match.matched

                    match_is_standalone = bool(
                        re.search(rf"\b{query_match}\b", input_text)
                    )
                except Exception:
                    match_is_standalone = False

                if not match_is_standalone:
                    flags.append(CommonFlag.IS_INSIDE_ANOTHER_WORD)

                # if match contains a /n in the middle (not as last or first character) reduce dist by the number of \n
                distance = match.dist
                match_has_newline = "\n" in match.matched
                if match_has_newline:
                    if len(match.matched) > 2:
                        middle_newlines = match.matched[1:-1].count("\n")
                        distance -= middle_newlines

                if isinstance(names[candidate], str):
                    matches.append(
                        FuzzyMatch(
                            start=match.start,
                            end=match.end,
                            matched=match.matched,
                            dist=distance,
                            possibility=candidate_str,
                            flags=flags,
                            origin=names[candidate]
                        )
                    )
                else:  # list or set
                    for val in names[candidate]:
                        matches.append(
                            FuzzyMatch(
                                start=match.start,
                                end=match.end,
                                matched=match.matched,
                                dist=distance,
                                possibility=candidate_str,
                                flags=flags,
                                origin=val
                            )
                        )

        batched_fuzzy_match_results.append(FuzzyMatchResult(matches))

    return batched_fuzzy_match_results
