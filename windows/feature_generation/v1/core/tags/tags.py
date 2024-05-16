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
# pylint: disable=line-too-long, invalid-name, redefined-builtin
# noqa: C0302
"""Core functions Similar to the ``flags.py`` but using tags.

Spark and parquet files are known
to suffer performance issues when there are too many columns. This sub-module
attempts to take advantage of pyspark's native ability to handle arrays. A user
has to decide when to use tags instead of flags, as flags are easier to handle.
"""

import logging
import operator as op
from functools import reduce
from typing import Callable, Dict, List, Union

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Window

from ...core.utils.arrays import _create_regex_from_list
from ...core.utils.arrays import array_rlike as arr_rlike
from ..utils.alias import alias

_PREFIX = "_tag_"
_NULL = "_null"

logger = logging.getLogger(__name__)


class CONSTANTS:  # pylint: disable=too-few-public-methods
    """Stores data type constants to be used internally."""

    TAG = "tag"
    VALUE = "value"


@alias()
def filter_tags(
    tags: Union[str, List[str]], tag_col_name: str = "tags"
) -> pyspark.sql.Column:
    """Helper function to filter a tag array.

    Use `alias` argument for setting the name of the output column.

    Args:
        tags: The tag or list of tags to filter the tag array column.
        tag_col_name: The name of the tag column. Defaults to "tags".

    Returns:
        A filtered array of tags as a pyspark column.
    """
    if isinstance(tag_col_name, pyspark.sql.Column):
        tag_col = (
            tag_col_name._jc.toString()  # pylint: disable=protected-access # noqa: SLF001
        )
    else:
        tag_col = tag_col_name

    if isinstance(tags, str):
        tags = [tags]

    tags = ",".join([f"'{x}'" for x in tags])

    return f.expr(f"filter({tag_col}, x -> x.{CONSTANTS.TAG} IN ({tags}))")


@alias()
def filter_rlike_tags(
    patterns: Union[str, List[str]], tag_col_name: str = "tags"
) -> pyspark.sql.Column:
    """Helper function to filter a tag array using rlike.

    Use `alias` argument for setting the name of the output column.

    Args:
        patterns: The patterns or list of patterns
        tag_col_name: The name of the tag column. Defaults to "tags".

    Returns:
        A filtered array of tags as a pyspark column.
    """
    if isinstance(tag_col_name, pyspark.sql.Column):
        tag_col = (
            tag_col_name._jc.toString()  # pylint: disable=protected-access # noqa: SLF001
        )
    else:
        tag_col = tag_col_name

    if isinstance(patterns, str):
        patterns = [patterns]

    patterns = _create_regex_from_list(patterns)
    return f.expr(f"filter({tag_col}, x -> x.{CONSTANTS.TAG} rlike '{patterns}')")


def contains_tags(
    tags: Union[str, List[str]], tag_col_name: Union[str, pyspark.sql.Column] = "tags"
) -> bool:
    """Returns True if the tag exists in the tag column."""
    return f.size(filter_tags(tags, tag_col_name)) > f.lit(0)


@alias()
def convert_tag_to_column(tag: str, tag_col_name: str) -> pyspark.sql.Column:
    """Extracts a tag from the tag array column.

    Use `alias` argument for setting the name of the output column.

    If there are duplicate tags, the first one will be taken. Best to ensure that
    there are no duplicate tags in the tag array function.
    """
    extracted_column = filter_tags(tags=tag, tag_col_name=tag_col_name).getItem(0)

    return extracted_column


@alias()
def extract_all_tag_names(
    tag_col_name: Union[str, pyspark.sql.Column] = "tags_all"
) -> pyspark.sql.Column:
    """Extracts all tag names from the tag column.

    Use `alias` argument for setting the name of the output column.
    """
    if isinstance(tag_col_name, pyspark.sql.Column):
        tag_col = (
            tag_col_name._jc.toString()  # pylint: disable=protected-access # noqa: SLF001
        )
    else:
        tag_col = tag_col_name

    transformed_tag_column = f.expr(f"transform({tag_col}, x -> x.{CONSTANTS.TAG})")

    return transformed_tag_column


@alias()
def extract_tag_value(
    tag: Union[str, pyspark.sql.Column], dtype: str = "double"
) -> pyspark.sql.Column:
    """Extracts the tag value.

    Note that tag values are currently converted to double.

    Use `alias` argument for setting the name of the output column.

    Args:
        tag: The column containing a tag.
        dtype: The expected dtype of the tag. Defaults to "double".

    Returns:
        A pyspark column.
    """
    tag_column = tag if isinstance(tag, pyspark.sql.Column) else f.col(tag)

    return tag_column.getField(CONSTANTS.VALUE).cast(dtype)


def contains_rlike_tags(
    patterns: Union[str, List[str]],
    tag_col_name: Union[str, pyspark.sql.Column] = "tags",
) -> bool:
    """Returns True if a tag exists in the tag column with the following patterns."""
    return f.size(filter_rlike_tags(patterns, tag_col_name)) > f.lit(0)


def construct_tag(
    tag: Union[str, pyspark.sql.Column],
    value: Union[int, float, pyspark.sql.Column] = 1,
) -> pyspark.sql.Column:
    """Constructs a tag, where a tag is defined by a tag name, a value, and a type.

    Note that the type represents the dtype of the value. The dtypes have been fixed
    to a few select values. This is to enable parsing out at a later stage.

    Args:
        tag: The name of the tag.
        value: The value to associate with the tag.

    Returns:
        A pyspark struct column.

    Raises:
        TypeError: If neither int nor float nor pyspark.sql.Column object is passed
            to ``value`` arg.
        TypeError: If neither string nor pyspark.sql.Column object is passed to ``tag``
            arg.
    """
    if isinstance(value, (float, int)):
        value_col = f.lit(value)
    elif isinstance(value, pyspark.sql.Column):
        value_col = value
    else:
        raise TypeError(
            "Expected either an int or float or pyspark.sql.Column object "
            "for ``value`` arg."
        )

    if isinstance(tag, str):
        tag_col = f.lit(tag)
    elif isinstance(tag, pyspark.sql.Column):
        tag_col = tag
    else:
        raise TypeError(
            "Expected either a str or pyspark.sql.Column object "
            "for ``tag`` arg."  # pylint: disable=implicit-str-concat
        )

    return f.struct(tag_col.alias(CONSTANTS.TAG), value_col.alias(CONSTANTS.VALUE))


def isin(input: str, values: List[str], tag: str) -> pyspark.sql.Column:
    """Adds tag as the column value if any of the list values is found in the column.

    Args:
        input: The input column name.
        values: A list of values to check the input column against.
        tag: The name of the tag.

    Returns:
        A spark column object
    """
    return (
        f.when(
            f.col(input).isin(values),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def rlike(input: str, values: List[str], tag: str) -> pyspark.sql.Column:
    """Adds tag as the column value if the regex pattern can be found in the column.

    Args:
        input: The input column name.
        values: A list of values to check the input column against.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    rlike_case = (
        f.when(
            f.col(input).rlike(_create_regex_from_list(values)),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )

    return rlike_case


def rlike_multi_col(
    inputs: List[Dict[str, str]], tag: str, operator: Callable = op.or_
) -> pyspark.sql.Column:
    """Creates a tag if any of the patterns matches for multiple columns.

    Args:
        inputs: A list of dictionaries containing `input` and `values`.
        tag: The name of the tag to generate.
        operator: The operator for multi columns. Defaults to `or_`.

    Returns:
        A spark column object

    Raises:
        ValueError: If input is not a list.
    """
    for cfg in inputs:
        if not isinstance(cfg["values"], list):
            raise ValueError(f"Expect values for input:`{cfg['input']}` to be list!")

    full_expr = reduce(
        operator,
        [
            f.col(input["input"]).rlike(_create_regex_from_list(input["values"]))
            for input in inputs
        ],
    )

    rlike_case = (
        f.when(
            full_expr,
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )

    return rlike_case


def dynamic_tag(expr: str, tag: str) -> pyspark.sql.Column:
    """Constructs a tag where the value of the tag is the evaluated ``expr``.

    Args:
        expr: The SQL expression.
        tag: The name of the tag.

    Returns:
        A spark column object
    """
    logger.warning("Ensure that dynamic tag '%s' expr returns a numeric column.", tag)
    return (
        f.when(
            f.expr(expr).isNotNull(),
            f.array(construct_tag(tag=tag, value=f.expr(expr))),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def one_hot_tag(expr: str, tag: str) -> pyspark.sql.Column:
    """Constructs a tag where the value is part of the tag name.

    Args:
        expr: The SQL expression.
        tag: The name of the tag.

    Returns:
        A spark column object
    """
    return (
        f.when(
            f.expr(expr).isNotNull(),
            f.array(
                construct_tag(
                    tag=f.concat(f.lit(tag), f.lit("_"), f.expr(expr)), value=1
                )
            ),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def expr_tag(expr: str, tag: str) -> pyspark.sql.Column:
    """Adds tag as the column value if expression evaluates to True.

    Args:
        expr: The SQL expression.
        tag: The name of the tag.

    Returns:
        A spark column object
    """
    return (
        f.when(
            f.expr(expr),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def tag_array_rlike(
    input: str, values: Union[str, List[str]], tag: str
) -> pyspark.sql.Column:
    """Generates a tag if any elements in the tag array regex match the pattern.

    Args:
        input: The input column name.
        values: A list of values to check the input column against using regex.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    return (
        f.when(
            contains_rlike_tags(patterns=values, tag_col_name=input),
            f.array(construct_tag(tag=tag, value=1)),
        ).otherwise(f.array(construct_tag(tag=_NULL)))
    ).alias(_PREFIX + tag)


def array_rlike(
    input: str, values: Union[str, List[str]], tag: str
) -> pyspark.sql.Column:
    """Generates a tag if any elements in the array regex match the pattern.

    Args:
        input: The input column name.
        values: A list of values to check the input column against using regex.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    return (
        f.when(
            f.size(arr_rlike(input=input, pattern=values)) > f.lit(0),
            f.array(construct_tag(tag=tag, value=1)),
        ).otherwise(f.array(construct_tag(tag=_NULL)))
    ).alias(_PREFIX + tag)


def tag_arrays_overlap(
    input: str, values: Union[str, List[str]], tag: str
) -> pyspark.sql.Column:
    """Adds tag as the column value if any of the list values is found in the tag array.

    Args:
        input: The input column name.
        values: A list of values to check the input column against.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    if isinstance(values, str):
        values = [values]

    return (
        f.when(
            contains_tags(tags=values, tag_col_name=input),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def arrays_overlap(
    input: str, values: Union[str, List[str]], tag: str
) -> pyspark.sql.Column:
    """Returns tag if any element in the list can be found in the array.

    Args:
        input: The input column name.
        values: A list of values to check the input column against.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    if isinstance(values, str):
        values = [values]

    value_array = f.array([f.lit(x) for x in values])

    return (
        f.when(
            f.arrays_overlap(input, value_array),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def tag_arrays_not_overlap(
    input: str,
    values: Union[str, List[str]],
    tag: str,
) -> pyspark.sql.Column:
    """Returns a tag if elements in values cannot be found in the tag array.

    Args:
        input: The input column name.
        values: A list of values to check the input column against.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    if isinstance(values, str):
        values = [values]

    return (
        f.when(
            ~contains_tags(tags=values, tag_col_name=input),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def arrays_not_overlap(
    input: str, values: Union[str, List[str]], tag: str
) -> pyspark.sql.Column:
    """Returns tag if any of the lsit values cannot be found in the array.

    Args:
        input: The input column name.
        values: A list of values to check the input column against.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    if isinstance(values, str):
        values = [values]

    value_array = f.array([f.lit(x) for x in values])

    return (
        f.when(
            ~f.arrays_overlap(input, value_array),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def tag_array_contains_all(
    input: str,
    values: Union[str, List[str]],
    tag: str,
) -> pyspark.sql.Column:
    """Returns a tag column if all the values can be found in the tag array.

    Pyspark's ``array_contains`` only allows to check for a single element, while
    ``arrays_overlap`` returns True if any element can be found.

    Args:
        input: The input column name.
        values: A list of values to check the input column against. Input column
            dtype should be array.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    if isinstance(values, str):
        values = [values]

    return (
        f.when(
            f.size(filter_tags(tags=values, tag_col_name=input)) == f.lit(len(values)),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def array_contains_all(
    input: str,
    values: Union[str, List[str]],
    tag: str,
) -> pyspark.sql.Column:
    """Returns tag if all elements in the list can be found in the array.

    Pyspark's ``array_contains`` only allows to check for a single element, while
    ``arrays_overlap`` returns True if any element can be found.

    Args:
        input: The input column name.
        values: A list of values to check the input column against. Input column
            dtype should be array.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    if isinstance(values, str):
        values = [values]

    intersect_tags = f.array([f.lit(x) for x in values])

    return (
        f.when(
            f.size(f.array_intersect(input, intersect_tags)) == f.lit(len(values)),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def tag_array_not_contains_all(
    input: str,
    values: Union[str, List[str]],
    tag: str,
) -> pyspark.sql.Column:
    """Returns tag if the tag array does not contain all the elements.

    Args:
        input: The input column name.
        values: A list of values to check the input column against. Input column
            dtype should be array.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    if isinstance(values, str):
        values = [values]

    return (
        f.when(
            ~(
                f.size(filter_tags(tags=values, tag_col_name=input))
                == f.lit(len(values))
            ),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def array_not_contains_all(
    input: str,
    values: Union[str, List[str]],
    tag: str,
) -> pyspark.sql.Column:
    """Returns a tag column if the array does not contain all the elements.

    Args:
        input: The input column name.
        values: A list of values to check the input column against. Input column
            dtype should be array.
        tag: The name of the tag.

    Returns:
        A spark column object.
    """
    if isinstance(values, str):
        values = [values]

    intersect_tags = f.array([f.lit(x) for x in values])

    return (
        f.when(
            ~(f.size(f.array_intersect(input, intersect_tags)) == f.lit(len(values))),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def count_and_compare_tags(
    tag: str,
    input: Union[str, pyspark.sql.Column],
    values: Union[str, List[str]],
    operator: Callable,
    y: Union[int, float],
) -> pyspark.sql.Column:
    """Add tag as the column value if the comparator evaluates to True.

    Counts the number of ``element`` within the ``input`` columns

    Args:
        tag: The name of the tag.
        input: The input column name, should be an array column.
        values: A tag or list of tags to filter to compare against.
        y: The number to compare against.
        operator: See https://docs.python.org/3/library/operator.html for standard
            operators available.

    Returns:
        A spark column object.
    """
    num_elements = f.size(filter_tags(tags=values, tag_col_name=input))

    return (
        f.when(
            operator(num_elements, y),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )


def nth_occurrence(
    tag: str,
    input: str,
    values: str,
    partition_by: Union[str, List[str]],
    order_by: Union[str, List[str]],
    n: int,
    descending: bool = False,
) -> pyspark.sql.Column:
    """Add tag as the column value if the tag appears as the nth time in the window.

    Args:
        tag: The name of the tag.
        input: The input column name, should be an array column.
        values: The tag to search for.
        partition_by: Partitioning column for window.
        order_by: Ordering column in the window.
        n: The nth time to tag.
        descending: Whether to reverse the order of the window.

    Returns:
        A spark column object.
    """
    if isinstance(partition_by, str):
        partition_by = [partition_by]
    if isinstance(order_by, str):
        order_by = [order_by]

    if descending:
        order_by = [f.col(x).desc() for x in order_by]

    # pylint: disable=invalid-name
    w = Window.partitionBy(partition_by).orderBy(order_by)  # noqa: E501
    # pylint: enable=invalid-name

    tag_exists = contains_tags(tags=values, tag_col_name=input).cast("int")

    return (
        f.when(
            tag_exists * f.sum(tag_exists).over(w) == f.lit(n),
            f.array(construct_tag(tag=tag, value=1)),
        )
        .otherwise(f.array(construct_tag(tag=_NULL)))
        .alias(_PREFIX + tag)
    )
