"""Data-integration transformations module."""

import inspect
import logging
import sys
from hashlib import md5
from typing import Dict, List

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    BinaryType,
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NullType,
    StringType,
    TimestampType,
)

logger = logging.getLogger(__name__)


def _get_datatype(python_type: str) -> DataType:
    """Helper function that chooses a Spark DataType based on a given python type.

    Accepted values for `python_type`: `str`, `bool`, `int`, `long`, `float`, `double`,
    `bytearray`, `bytes`, `decimal`, `date`, `datetime`, `time`, `timestamp`, `none`.

    If the `python_type` is not recognized, StringType will be returned.

    !!! note

        Complex types like MapType, ArrayType and StructType are not supported.

    Args:
        python_type: A Python type to be converted in `pyspark.sql.types`

    Returns:
        A `pyspark.sql.types` type

    Examples:
        >>> _get_datatype("int")
        IntegerType()
        >>> _get_datatype("str")
        StringType()
    """
    # mapping values taken from pyspark source code: `pyspark/sql/types.py`
    return {
        "str": StringType(),
        "bool": BooleanType(),
        "int": IntegerType(),
        "long": LongType(),
        "float": FloatType(),
        "double": DoubleType(),
        "bytearray": BinaryType(),
        "bytes": BinaryType(),
        "decimal": DecimalType(),
        "date": DateType(),
        "datetime": TimestampType(),
        "time": TimestampType(),
        "timestamp": TimestampType(),
        "none": NullType(),
    }.get(python_type, StringType())


def _assert_subset_in_df(df_columns: List[str], expected_columns: List[str]) -> None:
    """Check if a subset of columns is present in dataframe.

    Args:
        df_columns: List of columns of the dataframe
        expected_columns: List of columns to have their existence checked

    Examples:
        >>> _assert_subset_in_df(["a", "b"], ["c"])
        Assertion error with the message
        `Column(s) ['c'] is(are) not present in the dataframe...`
        >>> _assert_subset_in_df(["a", "b"], ["a"])
        No assertion error

    Raises:
        ValueError: Raised if one or more columns of `expected_columns` list is not
            present in the list of dataframe columns `df_columns`
    """
    # get only the expected cols in subset that are not present in the DF
    not_existing_cols = list(set(expected_columns) - set(df_columns))

    # if list is not empty
    if not_existing_cols:
        raise ValueError(
            f"Column(s) {not_existing_cols} is(are) not "
            "present in the dataframe. Interrupting transformation."
        )


def cast_df_columns(df: DataFrame, casting_map: dict) -> DataFrame:
    """Given a dataframe and a mapping dictionary, apply the intended castings.

    Args:
        df: The dataframe where the casting operation will be applied.
        casting_map: Casting operations to be applied to the dataframe.

    Returns:
        Dataframe with all the casting operations applied.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |       2 |
        >>> cast_df_columns(df, {"column_b": "str"}).collect()
        [Row(column_a=1, column_b='2')]
    """
    _assert_subset_in_df(df_columns=df.columns, expected_columns=casting_map.keys())

    # get a list with all columns of df, except the one being casted
    other_columns = df.columns
    new_casted_columns = []

    for column_name, new_column_type in casting_map.items():
        other_columns.remove(column_name)

        # convert python type into pyspark.sql.types object and append the selected col
        new_type = _get_datatype(new_column_type)
        new_casted_columns.append(f.col(column_name).cast(new_type).alias(column_name))

    # select other columns + the new one already casted
    # note: avoided .withColumn() to not hang on spark catalyst optimizer
    df = df.select(*other_columns, *new_casted_columns)

    return df
