from typing import Generator

import polars as pl

from data_structuring.components.readers.base_reader import BaseReader


class DataFrameReader(BaseReader):
    def __init__(self, dataframe: pl.DataFrame, data_column_name: str):
        """
        Initialize the DataFrameReader.
        Args:
            dataframe: A polars DataFrame containing the data to read.
            data_column_name: The name of the column to read values from.
        Raises:
            ValueError: If the specified column is not found in the DataFrame.
        """
        if data_column_name not in dataframe.columns:
            raise ValueError(
                f"Column '{data_column_name}' not found in DataFrame. "
                f"Available columns: {list(dataframe.columns)}"
            )
        self.data_column_name = data_column_name
        self.dataframe = dataframe.select(self.data_column_name)

    def read(self) -> Generator[str, None, None]:
        """
        Yield values from the specified DataFrame column
        Returns:
            Generator[str, None, None]: A generator yielding non-null values from the specified column as strings.
        """
        yield from self.dataframe[self.data_column_name].drop_nans().drop_nulls().cast(pl.String)
