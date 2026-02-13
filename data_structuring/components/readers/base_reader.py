"""
Module providing a base reader class for reading records from an input source.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

DEFAULT_ADDRESS_COLUMN = "address"
DEFAULT_SUGGESTED_COUNTRY_COLUMN = "suggested_country"
DEFAULT_FORCE_SUGGESTED_COUNTRY_COLUMN = "force_suggested_country"

# All values that are interpreted as "True" to enable the forced suggested country feature
# These values should all be in the same case (i.e.: upper/lower-case)
POSSIBLE_FORCED_FLAG_VALUES = [
    "true",
    "1",
    "yes",
    "y"
]


@dataclass(frozen=True, slots=True)
class AddressSample:
    """An address string with optional per-row metadata."""
    text: str
    suggested_country: str | None = None
    force_suggested_country: bool = False


class BaseReader(ABC):
    @abstractmethod
    def read(self) -> Generator[AddressSample, None, None]:
        """
        Abstract method to read records from an input source.
        Returns:
            Generator[AddressSample, None, None]: A generator yielding address samples from the input source.
        """
        raise NotImplementedError()
