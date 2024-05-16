# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organization
# and QuantumBlack, and any unauthorized use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organization with the prior written
# permission of QuantumBlack.
# pylint: disable=invalid-name
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


def _assert_columns_datatype(
    df: DataFrame, df_columns: List[str], expected_datatype: pyspark.sql.types
) -> None:
    """Check if a subset of columns belong to a given datatype.

    Args:
        df: The dataframe to have its columns datatypes validated.
        df_columns: List of columns of the dataframe to be validated.
        expected_datatype: Datatype that `df_columns` should match.

    Examples:
        >>> _assert_columns_datatype(df, df_columns=['text', 'num'], StringType)
        Some columns do not match expected datatype `StringType`: `['num']`"
        >>> _assert_columns_datatype(df, df_columns=['textA', 'textB'], StringType)
        None

    Raises:
        ValueError: Raised if one or more columns of `df_columns` list is not
            from datatype `expected_datatype` or if `expected_datatype` is not an
            instance of `pyspark.sql.types`.
    """
    # check for valid pyspark datatypes, showing the possible values in case of error
    ACCEPTED_CLASSES = tuple(  # pylint: disable=consider-using-generator
        [
            cls_obj
            for cls_name, cls_obj in inspect.getmembers(
                sys.modules["pyspark.sql.types"]
            )
            if inspect.isclass(cls_obj)
        ]
    )

    if not isinstance(expected_datatype, ACCEPTED_CLASSES):
        raise ValueError(
            "`expected_datatype` is not an instance of the accepted types: "
            f"{ACCEPTED_CLASSES}. Got {type(expected_datatype)}."
        )

    # get all columns that do not comply with the expected_datatype
    cols_not_from_expected_type = [
        (col_name, df.schema[col_name].dataType)
        for col_name in df_columns
        if not isinstance(df.schema[col_name].dataType, ACCEPTED_CLASSES)
    ]

    if cols_not_from_expected_type:
        raise ValueError(
            f"Some columns do not match expected datatypes `{ACCEPTED_CLASSES}`: "
            f"`{cols_not_from_expected_type}`"
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


def rename_df_columns(df: DataFrame, renaming_map: dict) -> DataFrame:
    """Given a dataframe and a mapping dictionary, apply the intended renames.

    Args:
        df: The dataframe where the rename operation will be applied.
        renaming_map: Renaming operations to be applied to the dataframe.

    Returns:
        Dataframe with all the renaming operations applied.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |       2 |
        >>> dict = {"column_a": "new_column_name"}
        >>> rename_df_columns(df, dict).collect()
        [Row(column_b=2, new_column_name=1)]
    """
    _assert_subset_in_df(df_columns=df.columns, expected_columns=renaming_map.keys())
    # get a list with all columns of df, except the one being renamed
    other_columns = df.columns
    new_renamed_columns = []

    for prev_column_name, new_column_name in renaming_map.items():
        other_columns.remove(prev_column_name)

        new_renamed_columns.append(f.col(prev_column_name).alias(new_column_name))

    # select other columns + the new one already renamed
    # note: avoided .withColumn() to not hang on spark catalyst optimizer
    df = df.select(*other_columns, *new_renamed_columns)

    return df


def trim_columns(df: DataFrame, trimming_cols: list) -> DataFrame:
    """Given a dataframe and a list of columns, apply trimming method on them.

    Args:
        df: The dataframe where the trim operation will be applied.
        trimming_cols: Columns to which trimming operations will be applied.

    Returns:
        Dataframe with all the trimming operations applied.

    Raises:
        ValueError: If some requested column to be trimmed is not of type `StringType`.

    Examples:
        Being `df`:

        >>> print(df)
        +---+------------------+----------+
        | id|           address|     state|
        +---+------------------+----------+
        |  1|  14851 Jeffrey Rd|    DE    |
        |  2|43421 Margarita St|        NY|
        |  3|  13111 Siemon Ave|        CA|
        +---+------------------+----------+
        >>> list = ["address", "state"]
        >>> trim_columns(df, list).show()
        +---+------------------+-----+
        | id|           address|state|
        +---+------------------+-----+
        |  1|  14851 Jeffrey Rd|   DE|
        |  2|43421 Margarita St|   NY|
        |  3|  13111 Siemon Ave|   CA|
        +---+------------------+-----+
    """
    # check if the cols in the list are valid
    _assert_subset_in_df(df_columns=df.columns, expected_columns=trimming_cols)

    # check if the cols in the list is from expected datatype (StringType)
    _assert_columns_datatype(
        df=df, df_columns=trimming_cols, expected_datatype=StringType
    )

    # get a list with all columns of df, except the ones being trimmed
    other_columns = df.columns
    new_trimmed_columns = []

    for col in trimming_cols:
        other_columns.remove(col)
        new_trimmed_columns.append(f.trim(col).alias(col))

    # select other columns + the new one already renamed
    # note: avoided .withColumn() to not hang on spark catalyst optimizer
    df = df.select(*other_columns, *new_trimmed_columns)

    return df


def trim_all_string_columns(df: DataFrame, instructions: None) -> DataFrame:
    # pylint: disable=unused-argument
    """Given a dataframe, trim all its `StringType` columns.

    This functions is just a wrapper around the `trim_columns` operations in all
    `StringType` columns in the dataframe. It is meant to make easier to trim all
    existing `StringType` columns from a dataframe, without the need to specify
    one-by-one, as trimming all textual columns is a common operation for
    cleansing texts.

    The `instruction` argument is set to None but it's not used. It just exists to
    comply with the predefined interface within `processor.transform()` method.

    Args:
        df: The dataframe where the trim operation will be applied.
        instructions: Not in use, exists to comply with the predefined interface
            with transformations.

    Returns:
        Dataframe with trimming operations applied to all `StringType` columns.

    Examples:
        Being `df`:

        >>> print(df)
        +---+------------------+----------+
        | id|           address|     state|
        +---+------------------+----------+
        |  1|  14851 Jeffrey Rd|    DE    |
        |  2|43421 Margarita St|        NY|
        |  3|  13111 Siemon Ave|        CA|
        +---+------------------+----------+
        >>> list = ["address", "state"]
        >>> trim_all_string_columns(df, list).show()
        +---+------------------+-----+
        | id|           address|state|
        +---+------------------+-----+
        |  1|  14851 Jeffrey Rd|   DE|
        |  2|43421 Margarita St|   NY|
        |  3|  13111 Siemon Ave|   CA|
        +---+------------------+-----+
    """
    # wrap the trim_columns() function, using all existing `StringType` columns
    stringtype_cols = [
        col_name
        for col_name in df.columns
        if isinstance(df.schema[col_name].dataType, StringType)
    ]
    logger.debug(
        "StringType columns identified inside `trim_all_string_columns`:\n%s",
        stringtype_cols,
    )

    # Execute existing `trim_columns` function, using the list of string columns
    trimmed_df = trim_columns(df=df, trimming_cols=stringtype_cols)

    return trimmed_df


def regex_replace_values(
    df: DataFrame,
    columns_dict: Dict[str, Dict[str, str]],
) -> DataFrame:
    r"""A function that replaces substrings in specified columns of a PySpark DataFrame.

    Based on regular expression patterns provided in a dictionary. The replacement
    can be partial (i.e. only replace matched substring) or full string replacement.

    Args:
        df (DataFrame): A PySpark DataFrame to be transformed
        columns_dict (dict): A dictionary with column names as keys and
            another dictionary as values, which specifies the regex pattern
            and replacement string for each pattern that should be replaced in
            the corresponding column.
            Example: {"col1": {"regex_pattern1": "replacement1",
            "regex_pattern2": "replacement2"}, "col2": {...}}

    Returns:
        DataFrame: A transformed PySpark DataFrame with replaced substrings based
        on the regex patterns provided.

    Raises:
        ValueError: If some requested column to be trimmed is not of type `StringType`
            or if the subset of columns are not present in the dataframe.

    Examples:
        >>> df.show()
        +---+-------+
        |_id|  label|
        +---+-------+
        |  1| email#|
        |  2|big_123|
        |  3|bigger$|
        |  4|bigger$|
        +---+-------+

        >>> columns_dict_full = {'label': {r".*big.*": '_FULL_',
                          r"[^a-zA-Z0-9_.-]":"_WILD_",
                          '\\d+': '_NUM_'}
               }
        >>> regex_replace_values(df, columns_dict_full).show()
        +---+-----------+
        |_id|      label|
        +---+-----------+
        |  1|email_WILD_|
        |  2|     _FULL_|
        |  3|     _FULL_|
        |  4|     _FULL_|
        +---+-----------+
        >>> columns_dict_partial = {'label': {r"big": '_PARTIAL_',
                          r"[^a-zA-Z0-9_.-]":"_WILD_",
                          '\\d+': '_NUM_'}
               }
        >>> regex_replace_values(df, columns_dict_partial).show()
        +---+------------------+
        |_id|             label|
        +---+------------------+
        |  1|       email_WILD_|
        |  2|   _PARTIAL___NUM_|
        |  3|_PARTIAL_ger_WILD_|
        |  4|_PARTIAL_ger_WILD_|
        +---+------------------+
    """
    # check if the cols in the dict are valid
    _assert_subset_in_df(df_columns=df.columns, expected_columns=columns_dict.keys())

    # check if the cols in the list is from expected datatype (StringType)
    _assert_columns_datatype(
        df=df, df_columns=columns_dict.keys(), expected_datatype=StringType
    )
    for column, regex_dict in columns_dict.items():
        for pattern, replacement in regex_dict.items():
            df = df.withColumn(
                column,
                f.when(
                    f.col(column).rlike(pattern),
                    f.regexp_replace(f.col(column), pattern, replacement),
                ).otherwise(f.col(column)),
            )
    return df


def fill_na(df: DataFrame, fill_na_map: dict) -> DataFrame:
    """Given a dataframe and a fill_na mapping, apply the intended fill_na operations.

    Args:
        df: The dataframe where the operation will be applied.
        fill_na_map: Dictionary containing a list of columns and related values to be
            applied when a null value is found.

    Returns:
        Dataframe with filled columns.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |    null |
        >>> dict = {"column_b": "hi"}
        >>> fill_na(df, dict).collect()
        [Row(column_a=1, column_b='hi')]
    """
    _assert_subset_in_df(df_columns=df.columns, expected_columns=fill_na_map.keys())

    # pyspark's `.fillna()` accepts a Dict[column, value] as its input
    df = df.fillna(fill_na_map)

    return df


def apply_sql_expression(df: DataFrame, expression_map: dict) -> DataFrame:
    """Given a dataframe and a mapping dictionary, apply the intended SQL expressions.

    Args:
        df: The dataframe where the expression will be applied.
        expression_map: Spark SQL expressions operations to be applied to the dataframe.

    Returns:
        Dataframe with all the Spark SQL expressions already applied.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b   |
        |-------- |---------- |
        |       1 |2020-10-10 |
        >>> dict = {"column_b": "date_format(column_b, 'y')"}
        >>> __apply_expression_to_column(df, dict).collect()
        [Row(column_a=1, column_b='2020')]
    """

    def __apply_expression_to_column(
        df: DataFrame, column_name: str, sql_expression: str
    ) -> DataFrame:
        """Apply a custom Spark SQL expression to a column of the dataframe.

        Args:
            df: The dataframe where the expression will be applied.
            column_name: The name of the column to apply the expression.
            sql_expression: The SparkSQL expression to be applied.

        Returns:
            Dataframe with the chosen column already created or replaced.

        Examples:
            Being `df`:

            >>> print(df)
            |column_a |column_b   |
            |-------- |---------- |
            |       1 |2020-10-10 |
            >>> __apply_expression_to_column(
                df, 'column_b', 'date_format(column_b, 'y')'
            ).collect()
            [Row(column_a=1, column_b='2020')]
        """
        # get a list with all columns of df, except the one being processed (if exists)
        other_columns = df.columns
        if column_name in other_columns:
            other_columns.remove(column_name)

        # select other columns + the new one
        # note: avoided .withColumn() to not hang on spark catalyst optimizer
        df = df.select(*other_columns, f.expr(sql_expression).alias(column_name))

        return df

    for column in expression_map.keys():
        df = __apply_expression_to_column(df, column, expression_map[column])

    return df


def drop_na_from_subset(df: DataFrame, subset: list) -> DataFrame:
    """Given a dataframe and a mapping dictionary, apply the intended dropna operation.

    Given a dataframe and a list of columns to look for, drop the rows whose values in
    the subset of columns are nulls.

    Args:
        df: The dataframe where the operation will be applied.
        subset: List of columns to look for rows containing null values.

    Returns:
        Dataframe with dropped rows.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |    null |
        |       2 |       2 |
        >>> drop_na_from_subset(df, ["column_b"]).collect()
        [Row(column_a=2, column_b=2)]
    """
    _assert_subset_in_df(df_columns=df.columns, expected_columns=subset)
    return df.na.drop(subset=subset)


def keep_columns(df: DataFrame, subset: list) -> DataFrame:
    """Given a dataframe and list of columns, keep them in the dataframe.

    Args:
        df: The dataframe where the select operation will be applied.
        subset: The list of columns to be kept.

    Returns:
        Dataframe with only the selected columns.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |    null |
        |       2 |       2 |
        >>> keep_columns(df, ["column_b"]).collect()
        [Row(column_b=None), Row(column_b=2)]
    """
    _assert_subset_in_df(df_columns=df.columns, expected_columns=subset)
    return df.select(*subset)


def drop_columns(df: DataFrame, subset: list) -> DataFrame:
    """Given a dataframe and list of columns, drop them in the dataframe.

    Args:
        df: The dataframe where the drop operation will be applied.
        subset: The list of columns to be dropped.

    Returns:
        Dataframe with dropped columns.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |    null |
        |       2 |       2 |
        >>> drop_columns(df, ["column_b"]).collect()
        [Row(column_a=1), Row(column_a=2)]
    """
    _assert_subset_in_df(df_columns=df.columns, expected_columns=subset)
    return df.drop(*subset)


def drop_duplicates(df: DataFrame, subset: list) -> DataFrame:
    """Given a dataframe and list of columns, drop duplicated rows in the dataframe.

    Args:
        df: The dataframe where the drop_duplicates operation will be applied.
        subset: The list of columns to be evaluated.

    Returns:
        Dataframe with dropped duplicated rows.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |    null |
        |       1 |       1 |
        |       2 |       2 |
        >>> drop_duplicates(df, ["column_a"]).collect()
        [Row(column_a=1, column_b=None),
         Row(column_a=2, column_b=2)]
    """
    _assert_subset_in_df(df_columns=df.columns, expected_columns=subset)
    return df.drop_duplicates(subset)


def _flag_matching_rows(df: DataFrame, expression_map: dict) -> DataFrame:
    """Given a dataframe and list of matching SQL expressions, flag all matching rows.

    Applies all Spark SQL Expressions present in the `expression_map`, creating new
    temporary columns. Each SQL Expression must return a True or Null value as they
    will be used to define if the row should be kept or dropped.

    !!! note

        The unmatching rows should have a `null` value if the SQL expression
        does not match and `true` if it matches. If these values are not used,
        unpredictable behaviour can be seen during the last filtering operation.

    Args:
        df: The dataframe where the flagging operation will be applied.
        expression_map: Spark SQL expressions operations dictionary to be applied to
            the dataframe in new temporary columns.

    Returns:
        Tuple containing both the original dataframe with some new temporary columns
        and the list of new columns to be evaluated/treated by the caller.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |       1 |
        |       2 |       2 |
        >>> dict = {"mark_gt1": "CASE WHEN column_a > 1 THEN True ELSE NULL END"}
        >>> _flag_matching_rows(df, dict)
        ([Row(column_a=1, column_b=1, mark_gt1_13a43d=None),
          Row(column_a=2, column_b=2, mark_gt1_13a43d=True),],
          ['mark_gt1_13a43d'])
    """
    # list of temporary columns that must be dropped after the operation
    temp_columns = []

    for filter_name, expression in expression_map.items():
        # create a new random named column for each new flagging column using hashing
        new_column_name = "_".join(
            (filter_name, md5(filter_name.encode()).hexdigest()[-6:])
        )

        # store these temp columns to allow the caller to remove them later
        temp_columns.append(new_column_name)
        df = df.select(
            "*",
            f.expr(expression).alias(new_column_name),
        )

    # returns tuple with the dataframe and additional column names (matching flags)
    return (df, temp_columns)


def _create_final_flag(df: DataFrame, expression_map: dict) -> DataFrame:
    """Create a resulting column flagging if temporary columns are true.

    Applies all Spark SQL Expressions present in the `expression_map`, creating a new
    resulting column that flags if the rows contains flags to be kept or dropped.

    Args:
        df: The dataframe from which rows will be deleted.
        expression_map: Spark SQL matching expressions dictionary,
            to be applied to the dataframe.

    Returns:
        Dataframe with filtered rows.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |       1 |
        |       2 |       2 |
        >>> dict = {"mark_gt1": "CASE WHEN column_a > 1 THEN True ELSE NULL END"}
        >>> _create_final_flag(df, dict)
        ([Row(column_a=1, column_b=1, mark_gt1_13a43d=None, final_flag_86f4de=None),
          Row(column_a=2, column_b=2, mark_gt1_13a43d=True, final_flag_86f4de=True),],
          ['mark_gt1_13a43d'],
          ['final_flag_86f4de'],)
    """
    # unpack the returned tuple
    df, temp_column_names = _flag_matching_rows(df=df, expression_map=expression_map)

    # apply a hashing strategy to create a new random named column, that will
    # store the final flag indicating if the each row must be dropped or not
    final_flag_column_name = "_".join(
        ("final_flag", md5("".join(temp_column_names).encode()).hexdigest()[-6:])
    )

    # convert the column names into a list of `pyspark.sql.Column` objects
    temp_columns = [f.col(x) for x in temp_column_names]
    # coalesce all the temp columns, resulting in a True value only if the temp columns
    # have some True in these columns
    df = df.select(
        "*",
        f.coalesce(*temp_columns).alias(final_flag_column_name),
    )

    # Return relevant information to the caller:
    # - the dataframe with original and additional temporary columns
    # - list of temporary column names to be dropped after filtering
    # - the name of the final column that must be used for filtering
    return (df, temp_column_names, final_flag_column_name)


def drop_rows(df: DataFrame, expression_map: dict) -> DataFrame:
    """Drop rows of a given dataframe if the final flag column is null.

    Args:
        df: The dataframe which rows will be deleted.
        expression_map: Spark SQL matching expressions, to be applied to the dataframe
            and have its rows deleted.

    Returns:
        Dataframe with dropped rows.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |       1 |
        |       2 |       2 |
        >>> dict = {"mark_gt1": "CASE WHEN column_a > 1 THEN True ELSE NULL END"}
        >>> drop_rows(df, dict).collect()
        [Row(column_a=1, column_b=1)]
        >>> Filter based on regular expression
        print(df)
        +---+------------------+-----+
        | id|           address|state|
        +---+------------------+-----+
        |  1|  14851 Jeffrey Rd|   DE|
        |  2|43421 Margarita St|   NY|
        |  3|  13111 Siemon Ave|   CA|
        +---+------------------+-----+
        >>> dict = {"mark_gt1": "CASE WHEN address like '%Rd'
                    THEN True ELSE NULL END"}
        >>> drop_rows(df, dict).show()
        +---+------------------+-----+
        | id|           address|state|
        +---+------------------+-----+
        |  2|43421 Margarita St|   NY|
        |  3|  13111 Siemon Ave|   CA|
        +---+------------------+-----+
    """
    df, temp_column_names, final_flag_column_name = _create_final_flag(
        df=df, expression_map=expression_map
    )

    # keep on the rows with an existing value in the final flag (that must be dropped)
    # filter (keep) only null values to keep everything that didn't match the conditions
    df = df.filter(f.col(final_flag_column_name).isNull())

    # add it to the list of columns to be dropped at the end
    # and drop all these temporary columns
    temp_column_names.append(final_flag_column_name)
    df = drop_columns(df=df, subset=temp_column_names)

    return df


def keep_rows(df: DataFrame, expression_map: dict) -> DataFrame:
    """Keep rows of a given dataframe if the final flag column is not null.

    Args:
        df: The dataframe which rows will be kept.
        expression_map: Spark SQL matching expressions, to be applied to the dataframe
            and have its rows deleted.

    Returns:
        Dataframe with dropped rows.

    Examples:
        Being `df`:

        >>> print(df)
        |column_a |column_b |
        |-------- |-------- |
        |       1 |       1 |
        |       2 |       2 |
        >>> dict = {"mark_gt1": "CASE WHEN column_a > 1 THEN True ELSE NULL END"}
        >>> keep_rows(df, dict).collect()
        [Row(column_a=2, column_b=2)]
        >>> Filter based on regular expression
        print(df)
        +---+------------------+-----+
        | id|           address|state|
        +---+------------------+-----+
        |  1|  14851 Jeffrey Rd|   DE|
        |  2|43421 Margarita St|   NY|
        |  3|  13111 Siemon Ave|   CA|
        +---+------------------+-----+
        >>> dict = {"mark_gt1": "CASE WHEN address like '%Rd' THEN True ELSE NULL END"}
        >>> keep_rows(df, dict).show()
        +---+----------------+-----+
        | id|         address|state|
        +---+----------------+-----+
        |  1|14851 Jeffrey Rd|   DE|
        +---+----------------+-----+
    """
    df, temp_column_names, final_flag_column_name = _create_final_flag(
        df=df, expression_map=expression_map
    )

    # keep only the rows with an existing value in the final flag (that must be kept)
    # filter (keep) only True values because they did match the expression conditions
    df = df.filter(f.col(final_flag_column_name).isNotNull())

    # add it to the list of columns to be dropped at the end
    # and drop all these temporary columns
    temp_column_names.append(final_flag_column_name)
    df = drop_columns(df=df, subset=temp_column_names)

    return df


def dataframe_sampling(
    df: DataFrame,
    fraction: float = 0.2,
    withReplacement: bool = False,
    seed: int = None,
) -> DataFrame:
    """Randomly sample a fraction of the rows in a PySpark DataFrame.

    Args:
        df (DataFrame): The PySpark DataFrame to sample from.
        fraction (float): Fraction of rows to generate, range [0.0, 1.0].
            Note that it does not guarantee to provide the exact number of
            the fraction of records.
        withReplacement (bool, optional): Whether to sample with replacement.
            Default is False, meaning that each row can be sampled only once.
            Basically, it allows you to produce duplicate rows as well.
        seed (int, optional): The random seed to use for reproducibility.
            Default is None, meaning that a random seed will be used.

    Returns:
        DataFrame: A PySpark DataFrame containing the sampled rows.

    Examples:
        Being `df`:
        >>> print(df)
        +---+--------+
        |_id|   label|
        +---+--------+
        |  1|  email#|
        |  1| big_123|
        |  2| bigger$|
        |  2|smaller$|
        |  3|  bigger|
        |  3|   sunny|
        |  4|   money|
        |  5|    lion|
        |  6|pressure|
        +---+--------+
        >>> df.sample(withReplacement=True, fraction=0.4, seed=35).show()
        +---+-------+
        |_id|  label|
        +---+-------+
        |  1|big_123|
        |  3| bigger|
        |  3| bigger|
        |  3| bigger|
        |  3| bigger|
        |  3| bigger|
        |  4|  money|
        +---+-------+
        >>> df.sample(withReplacement=False, fraction=0.4, seed=35).show()
        +---+--------+
        |_id|   label|
        +---+--------+
        |  1|  email#|
        |  2| bigger$|
        |  2|smaller$|
        |  3|  bigger|
        +---+--------+
    """
    return df.sample(withReplacement, fraction, seed)
