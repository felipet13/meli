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

# pylint: disable=line-too-long
"""Experimental module for handling pyspark array columns.

The functions in this module aggregate an array column to return a scalar value.

See: https://docs.databricks.com/_static/notebooks/higher-order-functions-tutorial-python.html
 for mode details.
"""  # noqa: E501

from typing import Callable, List

import pyspark
import pyspark.sql.functions as f

from ..utils.alias import alias


@alias()
def array_weighted_mean(value: str, weight: str) -> pyspark.sql.Column:
    """Returns the weighted mean given a weight array and a value array.

    Use `alias` argument for setting the name of the output column.

    Args:
        value: The array column representing the values.
        weight: The array column representing the weights.

    Returns:
        A pyspark numeric column.
    """
    numerator = f.expr(
        f"""
        AGGREGATE(
            ARRAYS_ZIP({weight}, {value}),
            CAST(0 AS DOUBLE),
            (acc, x) -> acc + x.{weight} * x.{value},
            acc -> acc
        )"""
    )

    denominator = f.expr(_sum(weight))

    weighted_mean = numerator / denominator

    return weighted_mean


@alias()
def array_variance(value: str, mean: str) -> pyspark.sql.Column:
    """Returns the variance of the array column.

    Note: The mean of the array has to be calculated and stored in a column before
    calling this function.
    Use `alias` argument for setting the name of the output column.

    Args:
        value: The array column of numeric type.
        mean: The column containing the mean of the array.

    Returns:
        A pyspark numeric column.
    """
    variance_expr = f.expr(
        f"""
        AGGREGATE(
            {value},
            CAST(0 AS DOUBLE),
            (acc, x) -> acc + x*x,
            acc -> (acc/SIZE({value})) - POW({mean}, 2)
        )"""
    )

    return variance_expr


@alias()
def array_stddev(value: str, mean: str) -> pyspark.sql.Column:
    """Returns the standard deviation of the array column.

    Note: The mean of the array has to be calculated and stored in a column before
    calling this function.

    Use `alias` argument for setting the name of the output column.

    Args:
        value: The array column of numeric type.
        mean: The column containing the mean of the array.

    Returns:
        A pyspark numeric column.
    """
    stddev_expr = f.expr(
        f"""
        AGGREGATE(
            {value},
            CAST(0 AS DOUBLE),
            (acc, x) -> acc + x*x,
            acc -> POW((acc/SIZE({value})) - POW({mean}, 2), 0.5)
        )"""
    )

    return stddev_expr


@alias()
def array_auc_trapezoidal(value: str, time: str) -> pyspark.sql.Column:
    """Calculates the AUC using the trapezoid rule.

    Use `alias` argument for setting the name of the output column.

    Args:
        value: The y-axis column.
        time: The x-axis column.

    Returns:
        A pyspark numeric column.
    """

    def _y_axis(val):
        return f"(element_at({val}, x) + element_at({val}, x+1))"

    def _x_axis(val):
        return f"(element_at({val}, x+1) - element_at({val}, x))"

    y_expr = _y_axis(value)
    x_expr = _x_axis(time)

    auc_trapezoidal_expr = f.expr(
        f"""
        AGGREGATE(
            SEQUENCE(1, size({value})-1),
            CAST(0 AS DOUBLE),
            (acc, x) -> acc + ({y_expr} * {x_expr}),
            acc -> acc * 0.5
        )
        """
    )

    return auc_trapezoidal_expr


def _mean(col: str):
    """Formula for mean using higher-order functions."""
    return f"""
        AGGREGATE(
            {col},
            CAST(0 AS DOUBLE),
            (acc, x) -> acc + x,
            acc -> acc/SIZE({col})
        )
        """


def _sum(col: str):
    """Formula for sum using higher-order functions."""
    return f"""
        AGGREGATE(
            {col},
            CAST(0 AS DOUBLE),
            (acc, x) -> acc + x,
            acc -> acc
        )
        """


@alias()
def array_min(value: str) -> pyspark.sql.Column:
    """Wrapper for native pyspark ``array_min`` for convenience.

    Use `alias` argument for setting the name of the output column.
    """
    min_in_array = f.array_min(value)

    return min_in_array


@alias()
def array_max(value: str) -> pyspark.sql.Column:
    """Wrapper for native pyspark ``array_max`` for convenience.

    Use `alias` argument for setting the name of the output column.
    """
    max_in_array = f.array_max(value)

    return max_in_array


@alias()
def array_auc_time_delta(value: str, time_delta: int) -> pyspark.sql.Column:
    """Calculates the AUC using the trapezoid rule with x-axis being parametrised.

    Use `alias` argument for setting the name of the output column.

    Args:
        value: The y-axis column.
        time_delta: The time delta scalar value.

    Returns:
        A pyspark numeric column.
    """
    auc_tdelta_expr = f.expr(
        f"""
        AGGREGATE(
            SEQUENCE(1, size({value})-1),
            CAST(0 AS DOUBLE),
            (acc, t)
                -> acc +
                ((ELEMENT_AT({value}, t) + ELEMENT_AT({value}, t+1)) * {time_delta}),
            acc -> acc * 0.5
        )
        """
    )
    return auc_tdelta_expr


@alias()
def array_sum(value: str):
    """Wrapper for pyspark higher order function sum for convenience.

    Use `alias` argument for setting the name of the output column.
    """
    ar_sum = f.aggregate(
        value,
        f.lit(0).cast("double"),
        lambda acc, x: acc + x,
    )
    return ar_sum


def _mean_acc(acc, x):
    """Formula for using array mean."""
    count = acc.count + 1
    total_sum = acc.sum + x
    return f.struct(total_sum.alias("sum"), count.alias("count"))


@alias()
def array_mean(value: str):
    """Wrapper for pyspark higher order function mean for convenience.

    Use `alias` argument for setting the name of the output column.
    """
    ar_mean = f.aggregate(
        value,
        f.struct(
            f.lit(0).cast("double").alias("sum"), f.lit(0).cast("int").alias("count")
        ),
        _mean_acc,
        lambda acc: acc.sum / acc.count,
    )
    return ar_mean


@alias()
def aggregate_over_slice(
    input_col: str,
    lower_bound: int,
    upper_bound: int,
    anchor: str,
    anchor_array: List[str],
    func: Callable,
) -> pyspark.sql.Column:
    """Aggregates values within a slice (window equivalent using arrays).

    Use `alias` argument for setting the name of the output column.
    Prerequisite is that the column contains an array of numeric types.

    Args:
        input_col: The array column containing the values to slice.
        lower_bound: The lower bound for the window, an integer column.
        upper_bound: The upper bound for the window, an integer column.
        anchor: The reference point for the upper and lower bounds, an integer column.
        anchor_array: Array column with all values from anchor column.
        func: Aggregation function to apply on the slice.

    Returns:
        A new pyspark array column filtered by the window specs.
    """
    filtered_window_expr = func(
        return_slice(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            anchor_array=anchor_array,
            value_array=input_col,
            anchor_col=anchor,
        )
    )

    return filtered_window_expr


@alias()
def return_slice(
    upper_bound: int,
    lower_bound: int,
    anchor_array: List[str],
    anchor_col: str,
    value_array: str,
) -> pyspark.sql.Column:
    """Returns slice of array given a relative rangebetween.

    Use `alias` argument for setting the name of the output column.

    Logic used for slice:
    syntax: slice(input_column, start index, length)
    anchor_position --> array_position of anchor in anchor_array col
    bound_difference --> abs(upper_bound - lower_bound)

    start_index: max(anchor_position + lower_bound, 1).
    Max is used to ensure start_index doesn't have negative values.
    length: If anchor_position + lower_bound + bound_difference <= 0 then 0,
    If anchor_position + upper_bound <= bound_difference
    then anchor_position + upper_bound,
    Otherwise bound_difference + 1

    Args:
        upper_bound: The upper bound for the window, an integer column.
        lower_bound: The lower bound for the window, an integer column.
        anchor_array: Array column with all values from anchor column.
        anchor_col: The reference point for upper and lower bounds, an integer column.
        value_array: Array column containing list of values.

    Returns:
        A new pyspark array column filtered by the window specs.
    """
    # plus 1 because we need to consider the current row as 0
    diff_lower_upper = abs(upper_bound - lower_bound) + 1

    # when the window slice is less than the starting array
    # array            |--------------|
    # window |------|
    # upper bound of window falls below array starting
    window_upper_bound = (
        f.col(anchor_col)
        + f.lit(lower_bound)
        + f.lit(
            diff_lower_upper - 1  # the minus 1 is crucial to get this to work, idk why
        )
    )

    condition1 = window_upper_bound < f.array_min(anchor_array)

    # array            |--------------|
    # window       |------|
    # overlap          |--|
    # when the window slice partially covers the starting
    condition2 = f.col(anchor_col) + f.lit(lower_bound) < f.array_min(anchor_array)

    # array            |--------------|
    # window                             |------|
    # when the window slice exceeds the array
    condition3 = f.col(anchor_col) + f.lit(lower_bound) > f.array_max(anchor_array)

    # returns 0 if not in array
    # need to use expr here for array position to blend column and lits
    # otherwise get Column not Iterable error
    array_pos = f.expr(
        f"""
        array_position(
            {anchor_array},
            {anchor_col} + {lower_bound}
        )
        """
    )

    # since we exceed, return any number greater than size of array
    # could potentially return 99999 since what are the odds?
    # slice function can handle where starting position is beyond the last element
    start_pos = f.when(condition3, f.size(anchor_array) + 1).otherwise(
        # if return 0, default to 1 so function does not error
        f.greatest(array_pos, f.lit(1))
    )

    length = (
        f.when(condition1, f.lit(0))
        .when(
            # when the window slice partially covers the starting, return overlap
            condition2,
            (
                (f.col(anchor_col) + f.lit(lower_bound) + f.lit(diff_lower_upper))
                - f.array_min(anchor_array)
            ),
        )
        .otherwise(f.lit(diff_lower_upper))
    )

    slice_expr = f.slice(value_array, start_pos, length)
    return slice_expr
