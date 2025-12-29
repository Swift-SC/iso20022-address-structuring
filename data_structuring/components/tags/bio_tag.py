"""
Module providing the BIO tag class.
"""
from enum import StrEnum
from typing import List

from pydantic import BaseModel

from data_structuring.components.tags.tag import Tag


class BIO(StrEnum):
    """
    Represents the possible values for the BIO tags (Begin, Inside, or Other).
    """

    BEFORE = "B-"
    INSIDE = "I-"
    OTHER = "OTHER"


class BIOTag(BaseModel, frozen=True, use_enum_values=True):
    """
    Represents a BIO Tag (Begin, Inside, or Other).
    A BIO tag is used to identify the start and end of a entity that can be composed of multiple tokens/words.
    Example:
        Sentence: ["I", "live", "in", "new", "york"]
        Tags    : ["B-PER", "OTHER", "OTHER", "B-LOC", "I-LOC"]
        Where `PER` stands for Person, `LOC` stands for Location, and `O` stands for Other.
    """

    tag: Tag
    bio: BIO

    @classmethod
    def create_other(cls) -> "BIOTag":
        return cls(tag=Tag.OTHER, bio=BIO.OTHER)

    @classmethod
    def create_before(cls, tag: Tag) -> "BIOTag":
        return cls(tag=tag, bio=BIO.BEFORE)

    @classmethod
    def create_inside(cls, tag: Tag) -> "BIOTag":
        return cls(tag=tag, bio=BIO.INSIDE)

    @classmethod
    def create_all(cls, tag: Tag) -> List["BIOTag"]:
        return [
            cls.create_before(tag),
            cls.create_inside(tag),
        ]

    def __str__(self) -> str:
        return f"{self.bio.value}{self.tag.value}"

    def __hash__(self):
        return hash((self.tag, self.bio))
