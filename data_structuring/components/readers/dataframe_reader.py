from typing import Generator

import polars as pl

from data_structuring.components.readers.base_reader import BaseReader, AddressSample, POSSIBLE_FORCED_FLAG_VALUES, \
    DEFAULT_SUGGESTED_COUNTRY_COLUMN, DEFAULT_FORCE_SUGGESTED_COUNTRY_COLUMN


class DataFrameReader(BaseReader):
    def __init__(self,
                 dataframe: pl.DataFrame | pl.LazyFrame,
                 data_column_name: str,
                 suggested_country_column: str | None = None,
                 force_suggested_country_column: str | None = None):
        """
        Initialize the DataFrameReader.
        Args:
            dataframe: A polars DataFrame or LazyFrame containing the data to read.
            data_column_name: The name of the column to read values from.
            suggested_country_column: Optional column with suggested country codes (ISO 2-letter).
            force_suggested_country_column: Optional column with boolean flags to force the suggested country.
        Raises:
            ValueError: If the specified data column is not found in the DataFrame.
        """
        # Ensure the dataframe is a LazyFrame
        self.dataframe = dataframe.lazy()
        # Collect dataframe column names
        columns_in_df = self.dataframe.collect_schema().names()
        if data_column_name not in columns_in_df:
            raise ValueError(
                f"Column '{data_column_name}' not found in DataFrame. "
                f"Available columns: {list(columns_in_df)}"
            )
        self.data_column_name = data_column_name
        self.suggested_country_column = (suggested_country_column
                                         if suggested_country_column in columns_in_df
                                         else None)
        self.force_suggested_country_column = (force_suggested_country_column
                                               if force_suggested_country_column in columns_in_df
                                               else None)

        columns = [self.data_column_name]
        if self.suggested_country_column:
            columns.append(self.suggested_country_column)
        if self.force_suggested_country_column:
            columns.append(self.force_suggested_country_column)
        self.dataframe = self.dataframe.select(columns)

    def read(self) -> Generator[AddressSample, None, None]:
        """
        Yield AddressSample objects from the specified DataFrame columns.
        Returns:
            Generator[AddressSample, None, None]: A generator yielding address samples.
        """
        filtered = self.dataframe.drop_nulls(self.data_column_name)
        if self.suggested_country_column:
            filtered = filtered.with_columns(
                (pl.col(self.suggested_country_column).cast(pl.String)
                 .str.strip_chars().str.to_uppercase())
            ).with_columns(
                (pl.when((pl.col(self.suggested_country_column).str.len_chars() == 0))
                 .then(None)
                 .otherwise(pl.col(self.suggested_country_column))
                 .name.keep())
            )
        else:
            self.suggested_country_column = DEFAULT_SUGGESTED_COUNTRY_COLUMN
            filtered = filtered.with_columns(
                pl.lit(None).alias(self.suggested_country_column)
            )

        if self.force_suggested_country_column:
            filtered = filtered.with_columns(
                (pl.col(self.force_suggested_country_column).cast(pl.String)
                 .str.to_lowercase().str.strip_chars().is_in(POSSIBLE_FORCED_FLAG_VALUES).fill_null(False))
            )
        else:
            self.force_suggested_country_column = DEFAULT_FORCE_SUGGESTED_COUNTRY_COLUMN
            filtered = filtered.with_columns(
                pl.lit(False).alias(self.force_suggested_country_column)
            )

        filtered = filtered.with_columns(
            address_samples=(
                pl.struct(pl.all())
                .map_elements(lambda row:
                              AddressSample(
                                  text=row[self.data_column_name],
                                  suggested_country=row[self.suggested_country_column],
                                  force_suggested_country=row[self.force_suggested_country_column]
                              ), return_dtype=pl.Object)
            )
        )

        yield from filtered.select("address_samples").collect().to_series(0).to_list()
