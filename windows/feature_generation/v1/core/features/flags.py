# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.
# pylint: disable=invalid-name, redefined-builtin
"""A function that takes a spark dataframe and creates flag columns.

Usually when creating flags for features, a lot of ``IF...ELSE`` or
``CASE WHEN`` are needed. By abstracting common operations, this reduces the need for
massive ``IF...ELSE`` statements, and with the added benefit that a non-technical
person can verify the values corresponding to a certain flag. Several default
functions are provided (``rlike``, ``isin``, and ``rlike_extract``), or the user is
free to provide their own flag function.
"""

from typing import Dict, List, Optional, Union

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Window

from ...core.tags.tags import contains_tags
from ...core.utils.arrays import _create_regex_from_list


def rlike(
    input: str,
    values: Union[str, List[str]],
    output: str,
    return_value: Optional[str] = None,
) -> pyspark.sql.Column:
    """Creates a 1 or 0 flag if the regex pattern can be found in the column.

    Args:
        input: Input column name
        values: A list of values to check the input column against
        output: Output column name
        return_value: The string name of the column to return. Defaults
            to return `f.lit(1)`.

    Returns:
        A spark column object
    """
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        raise TypeError("Expected a list or string for ``values`` arg.")

    return_value = f.col(return_value) if return_value else f.lit(1)

    rlike_case = (
        f.when(f.col(input).rlike(_create_regex_from_list(values)), return_value)
        .otherwise(0)
        .alias(output)
    )

    return rlike_case


def isin(
    input: str,
    values: List[str],
    output: str,
    return_value: Optional[str] = None,
) -> pyspark.sql.Column:
    """Creates a 1 or 0 flag if any of the list values is found in the string column.

    Args:
        input: Input column name
        values: A list of values to check the input column against
        output: Output column name
        return_value: The string name of the column to return. Defaults
            to return `f.lit(1)`.

    Returns:
        A spark column object
    """
    return_value = f.col(return_value) if return_value else f.lit(1)

    isin_case = (
        f.when(f.col(input).isin(values), return_value).otherwise(0).alias(output)
    )

    return isin_case


def arrays_overlap(
    input: str,
    values: Union[str, List[str]],
    output: str,
    return_value: Optional[str] = None,
) -> pyspark.sql.Column:
    """Creates a 1 or 0 flag if any of the list values is found in the array column.

    Args:
        input: Input column name
        values: A list of values to check the input column against
        output: Output column name
        return_value: The string name of the column to return. Defaults
            to return `f.lit(1)`.

    Returns:
        A spark column object
    """
    return_value = f.col(return_value) if return_value else f.lit(1)

    if isinstance(values, str):
        values = [values]

    value_array = f.array([f.lit(x) for x in values])

    overlap_case = (
        f.when(f.arrays_overlap(input, value_array), return_value).otherwise(f.lit(0))
    ).alias(output)

    return overlap_case


def tag_arrays_overlap(
    input: str,
    values: Union[str, List[str]],
    output: str,
    return_value: Optional[str] = None,
) -> pyspark.sql.Column:
    """Creates a 1 or 0 flag if any of the list values is found in the tag array column.

    Args:
        input: Input column name
        values: A list of values to check the input column against
        output: Output column name
        return_value: The string name of the column to return. Defaults
            to return `f.lit(1)`.

    Returns:
        A spark column object
    """
    return_value = f.col(return_value) if return_value else f.lit(1)

    if isinstance(values, str):
        values = [values]

    overlap_case = (
        f.when(contains_tags(tags=values, tag_col_name=input), return_value).otherwise(
            f.lit(0)
        )
    ).alias(output)

    return overlap_case


def rlike_extract(
    input: str,
    values: List[str],
    output: str,
    suffix: Optional[str] = None,
    return_value: Optional[str] = None,
) -> List[pyspark.sql.Column]:
    """Creates two columns.

        * a 1 or 0 flag if the if the regex pattern can be found in the column.
        * column containing regex matched word from string in column

    Args:
        input: Input column name
        values: A list of values to check the input column against
        output: Output column name
        suffix: Output column suffix for column returning the matched word
        return_value: The string name of the column to return. Defaults
            to return `f.lit(1)`.

    Returns:
        A list of spark column object(s)
    """
    if suffix is None:
        matched_value_column_name = output + "_matched_value"
    else:
        matched_value_column_name = output + suffix

    rlike_case = rlike(
        input=input, values=values, output=output, return_value=return_value
    )
    matched_word = regexp_extract(
        input=input, values=values, output=matched_value_column_name
    )

    return [rlike_case, matched_word]


def regexp_extract(
    input: str, values: Union[str, List[str]], output: str
) -> pyspark.sql.Column:
    """Column containing regex matched word from string in column.

    Args:
        input: The name of the input column.
        values: A list of values to check the input column against.
        output: The name of the output column.

    Returns:
        A spark column object
    """
    if isinstance(values, str):
        values = [values]

    matched_word = f.regexp_extract(f.col(input), _create_regex_from_list(values), 0)
    case_empty_string = (
        f.when(matched_word != f.lit(""), matched_word)
        .otherwise(f.lit(None).astype("string"))
        .alias(output)
    )

    return case_empty_string


def nth_occurrence(
    output: str,
    input: str,
    values: str,
    window: Dict[str, Union[str, List[str]]],
    n: int,
    return_value: Optional[str] = None,
) -> pyspark.sql.Column:
    """Add tag as column value if the tag appears as the nth time in the given window.

    Args:
        output: The output column name.
        input: The input column name, should be an array column.
        values: The tag to search for.
        window: The window specification to search in.
        n: The nth time to look for.
        return_value: The string name of the column to return. Defaults
            to return `f.lit(1)`.

    Returns:
        A spark column object.
    """
    return_value = f.col(return_value) if return_value else f.lit(1)

    partition_by = window.get("partition_by")
    order_by = window.get("order_by")

    if isinstance(partition_by, str):
        partition_by = [partition_by]
    if isinstance(order_by, str):
        order_by = [order_by]

    # pylint: disable=invalid-name
    w = Window.partitionBy(partition_by).orderBy(order_by)  # noqa: E501
    # pylint: enable=invalid-name

    tag_exists = contains_tags(tags=values, tag_col_name=input).cast("int")

    return (
        f.when(tag_exists * f.sum(tag_exists).over(w) == f.lit(n), return_value)
        .otherwise(f.lit(0))
        .alias(output)
    )


def null(input: str, values: bool, output: str) -> pyspark.sql.Column:
    """If values is True, whether column is null; Otherwise, whether it's false.

    Args:
        input: input column
        values: bool
        output: output column

    Returns:
        A spark column object
    """
    if values:
        spark_col = f.when(f.col(input).isNull(), f.lit(1)).otherwise(0).alias(output)
    else:
        spark_col = (
            f.when(f.col(input).isNotNull(), f.lit(1)).otherwise(0).alias(output)
        )

    return spark_col


def expr_col(expr: str, output: str) -> pyspark.sql.Column:
    """Create a feature column from expression.

    Args:
        expr: The expression to make a feature column
        output: The name of column to have the feature

    Returns:
        The column to have a feature value created by the given expression
    """
    return f.expr(expr).alias(output)


def expr_flag(
    expr: str, output: str, return_value: Optional[str] = None
) -> pyspark.sql.Column:
    """Create a flag column from expression.

    Args:
        expr: The expression to make a flag column
        output: The name of column to have the flag
        return_value: The string name of the column to return. Defaults
            to return `f.lit(1)`.

    Returns:
        The column to have a flag created by the given expression
    """
    return_value = f.col(return_value) if return_value else f.lit(1)

    return f.when(f.expr(expr), return_value).otherwise(f.lit(0)).alias(output)


def coalesce(
    output: str,
    cols: List[str],
) -> pyspark.sql.Column:
    """Create a feature column from coalesce.

    Args:
        output: The name of column to have the feature
        cols: The column names to coalesce

    Returns:
        The column to have a feature value created by the given coalesce statement
    """
    return f.coalesce(*[f.col(column) for column in cols]).alias(output)
