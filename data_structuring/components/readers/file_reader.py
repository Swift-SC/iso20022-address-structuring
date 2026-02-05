from pathlib import Path
from typing import Generator

import polars as pl

from data_structuring.components.readers.base_reader import BaseReader, AddressSample
from data_structuring.components.readers.dataframe_reader import DataFrameReader


class TextFileReader(BaseReader):
    def __init__(self, file_path: Path | str):
        self.file_path = file_path

    def read(self) -> Generator[AddressSample, None, None]:
        """Open the text file and yield each line until EOF."""
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Yield raw line content without trailing newline
                yield AddressSample(text=line.rstrip("\n"))


class CsvFileReader(BaseReader):
    def __init__(self,
                 file_path: Path | str,
                 data_column_name: str,
                 sep: str = ",",
                 encoding: str = "utf8",
                 suggested_country_column: str | None = None,
                 force_suggested_country_column: str | None = None):
        self.file_path = file_path
        self.data_column_name = data_column_name
        self.sep = sep
        self.encoding = encoding
        self.suggested_country_column = suggested_country_column
        self.force_suggested_country_column = force_suggested_country_column

    def read(self) -> Generator[AddressSample, None, None] | None:
        """Stream values from a CSV column lazily.

        Yields AddressSample objects with optional suggested_country and
        force_suggested_country metadata when the corresponding columns exist.
        """
        try:
            for chunk in pl.scan_csv(
                    self.file_path,
                    separator=self.sep,
                    encoding=self.encoding,
                    infer_schema=False
            ).collect_batches(
                chunk_size=10000,
                maintain_order=True,
                lazy=True,
                engine="streaming"
            ):
                return DataFrameReader(
                    dataframe=chunk,
                    data_column_name=self.data_column_name,
                    suggested_country_column=self.suggested_country_column,
                    force_suggested_country_column=self.force_suggested_country_column
                ).read()

        except pl.exceptions.ColumnNotFoundError as e:
            raise ValueError(f"Column '{self.data_column_name}' not found in CSV file: {self.file_path}") from e
