"""
Module providing a text stream reader class for reading records from a TextIO stream.
"""
from typing import Generator, TextIO
from data_structuring.components.readers.base_reader import BaseReader, AddressSample


class TextStreamReader(BaseReader):
    def __init__(self, text_stream: TextIO):
        self.text_stream = text_stream

    def read(self) -> Generator[AddressSample, None, None]:
        """
        Yield AddressSample objects one by one from a TextIO stream until EOF.
        Args:
            None - uses the TextIO object provided at initialization.

        Yields:
            AddressSample for each line from the stream without trailing newlines.
        """
        for line in self.text_stream.readlines():
            # Yield raw line content without trailing newline
            yield AddressSample(text=line.rstrip("\n"))
