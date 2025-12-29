"""
Module providing the post code matching classes.
"""
from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, RootModel, NonNegativeInt


class PostcodeMatch(BaseModel):
    # Postcodematch-related attributes
    start: NonNegativeInt
    end: NonNegativeInt
    matched: str
    possibility: str
    origin: str


# Wrapper around List[PostcodeMatch]
class PostcodeMatchResult(RootModel[List[PostcodeMatch]]):

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, idx):
        return self.root[idx]

    def __len__(self):
        return len(self.root)

    @classmethod
    def merge(cls, r1: PostcodeMatchResult, r2: PostcodeMatchResult) -> PostcodeMatchResult:
        return cls(r1.root + r2.root)


def find_postcode_town_matches(postcodes_dict: dict, regex_list: List[str], text: str,
                               postcode_regex_structure: str = ""):
    processed_text = re.sub(re.compile('[^A-Z0-9]'), ' ', text)
    towns_matched = []
    for regex in regex_list:
        for match in re.finditer(re.compile(regex + postcode_regex_structure), processed_text):
            dict_match = next((x for x in re.findall(regex, match.group(0))), None)
            if dict_match in postcodes_dict:
                for entry in postcodes_dict[dict_match]:
                    towns_matched.append(
                        PostcodeMatch(
                            start=match.start(),
                            end=match.end(),
                            matched=match.group(0),
                            possibility=entry[0],
                            origin=entry[1]
                        ))
    return PostcodeMatchResult(towns_matched)
