"""
Module providing a base reader class for reading records from an input source.
"""
from abc import ABC, abstractmethod
from typing import Generator


class BaseReader(ABC):
    @abstractmethod
    def read(self) -> Generator[str, None, None]:
        """
        Abstract method to read records from an input source.
        Returns:
            Generator[str, None, None]: A generator yielding records from the input source.
        """
        raise NotImplementedError()
