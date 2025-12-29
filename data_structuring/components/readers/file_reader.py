from pathlib import Path
from typing import Generator

import polars as pl

from data_structuring.components.readers.base_reader import BaseReader


class TextFileReader(BaseReader):
    def __init__(self, file_path: Path | str):
        self.file_path = file_path

    def read(self) -> Generator[str, None, None]:
        """Open the text file and yield each line until EOF."""
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Yield raw line content without trailing newline
                yield line.rstrip("\n")


class CsvFileReader(BaseReader):
    def __init__(self, file_path: Path | str, data_column_name: str, sep: str = ",", encoding: str = "utf8"):
        self.file_path = file_path
        self.data_column_name = data_column_name
        self.sep = sep
        self.encoding = encoding

    def read(self) -> Generator[str, None, None]:
        """Stream values from a CSV column lazily.

        Yields non-null values from the specified column as strings.
        """
        try:
            for chunk in pl.scan_csv(
                    self.file_path,
                    separator=self.sep,
                    encoding=self.encoding,
                    infer_schema=False
            ).select(self.data_column_name).collect_batches(
                # Split into batches to allow lazy loading of the column
                chunk_size=10000,
                maintain_order=True,
                lazy=True,
                engine="streaming"
            ):
                # Drop NaNs and yield as strings
                yield from chunk[self.data_column_name].drop_nans().drop_nulls().cast(pl.String)
        except ValueError as e:
            # Provide a clearer error when the column is missing
            if "Usecols" in str(e) and self.data_column_name in str(e):
                raise ValueError(f"Column '{self.data_column_name}' not found in CSV file: {self.file_path}") from e
            raise
