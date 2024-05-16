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

The functions in this module transforms an array column to return another array column.

See: https://docs.databricks.com/_static/notebooks/higher-order-functions-tutorial-python.html
 for mode details.
"""  # noqa: E501

from typing import Optional, Union

# pylint: disable=invalid-name
import pyspark
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, FloatType
from scipy.interpolate import interp1d

from ..utils.alias import alias


@alias()
def array_distinct(value: str) -> pyspark.sql.Column:
    """Wrapper for native pyspark ``array_distinct`` for convenience.

    Use `alias` argument for setting the name of the output column.
    """
    distinct = f.array_distinct(value)

    return distinct


@alias()
def array_smooth_ts_values(value: str, length: int) -> pyspark.sql.Column:
    """Smooths the values.

    Use `alias` argument for setting the name of the output column.

    Args:
        value: The y-axis column.
        length: The unit range interval to consider(i.e if range = 3, the time diff
            would be 3*time_delta)

    Returns:
        A pyspark string column.
    """

    def smooth(val, trange):
        return f"slice({val}, t, {trange})"

    smooth_ts_expr = f.expr(
        f"""
        TRANSFORM(
            SEQUENCE(1, size({value})),
            t -> AGGREGATE(
                {smooth(value, length)},
                CAST(0 AS DOUBLE),
                (acc, x) -> acc + x,
                acc -> ROUND(acc/{length}, 2)
                )
            )

        """
    )

    return smooth_ts_expr


@alias()
def array_derivative(value: str, time_delta: int) -> pyspark.sql.Column:
    """Calculates the first derivatives of data.

    Use `alias` argument for setting the name of the output column.

    Args:
        value: The value column.
        time_delta: The time difference between each reading.

    Returns:
        A pyspark column.
    """

    def _derivative(val, delta):
        return f"""
        case when t == size({val}) then 0
        else (element_at({val}, t+1) - element_at({val}, t))/ {delta} end
        """

    derivative_expr = f.expr(
        f"""
        TRANSFORM(
            SEQUENCE(1, size({value})),
            t -> ROUND({_derivative(value, time_delta)}, 2)
            )

        """
    )

    return derivative_expr


@alias()
def array_local_peaks_from_derivative(value: str) -> pyspark.sql.Column:
    """Finds local peaks based on first derivative values.

    Use `alias` argument for setting the name of the output column.

    The output column is an array flagging local peaks. A minimum peak is flagged with
    -1 and a maximum peak is flagged with 1.

    Args:
        value: The y-axis column.

    Returns:
        A pyspark column.
    """

    def _peaks(val):
        return f"""
                CASE WHEN t == 1 THEN 0
                WHEN (element_at({val},t-1) >= 0 AND element_at({val}, t) < 0) THEN 1
                WHEN (element_at({val}, t-1) < 0 AND element_at({val}, t) >= 0) THEN -1
                ELSE 0 END
                """

    peaks_expr = f.expr(
        f"""
        TRANSFORM(
            SEQUENCE(1, size({value})),
            t ->  {_peaks(value)}
            )

        """
    )

    return peaks_expr


@alias()
def scipy_interpolate(
    spine_index: str,
    time_index: str,
    value: str,
    kind: Union[str, int],
) -> pyspark.sql.Column:
    """Interpolates values based on scipy `interp1d` function.

    Use `alias` argument for setting the name of the output column.

    Args:
        spine_index: The spine column name.
        time_index: The time index column name.
        value: The y-axis column name.
        kind: Specifies the kind of interpolation as a string or as an integer
            specifying the order of the spline interpolator to use. More details in
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html.

    Returns:
        A pyspark column with the value column filled for the spine.
    """  # noqa: E501

    # Define scipy udf. Kind is defined externally to be able to pass as a string.
    @f.udf(returnType=ArrayType(FloatType()))
    def _scipy_interpolate_udf(spine_index, time_index, value):
        _f = interp1d(time_index, value, kind)
        ynew = _f(spine_index)
        return ynew.tolist()

    interpolated = _scipy_interpolate_udf(spine_index, time_index, value)

    return interpolated


@alias()
def forward_fill(spine_index: str, time_index: str, value: str) -> pyspark.sql.Column:
    """Forward fills values for a given spine.

    Use `alias` argument for setting the name of the output column.

    Args:
        spine_index: The spine column name.
        time_index: The time index column name.
        value: The y-axis column name.

    Returns:
        A pyspark column with the value column filled for the spine.
    """  # noqa: E501

    # Define forward_fill udf.
    @f.udf(returnType=ArrayType(FloatType()))
    def _forward_fill(spine_index, time_index, value):
        lookup = {}
        for t, v in zip(time_index, value):
            lookup[t] = v

        current_val = 0
        result = []
        for s in spine_index:
            _v = lookup.get(s, current_val)
            result.append(_v)
            current_val = _v
        return result

    forward_filled = _forward_fill(spine_index, time_index, value)

    return forward_filled


@alias()
def interpolate_constant(
    spine_index: str,
    time_index: str,
    value: str,
    constant: Optional[float] = None,
) -> pyspark.sql.Column:
    """Interpolate using constant value for a given spine.

    Use `alias` argument for setting the name of the output column.

    Args:
        spine_index: Column containing all spine date indices in an array.
        time_index: Time index array column.
        value: Array column containing the values to be aggregated.
        constant: Value to be used for interpolation.

    Returns:
        A pyspark array of struct column interpolated with input values.
    """
    interpolate_val = float(constant) if constant is not None else constant

    @f.udf(returnType=ArrayType(FloatType()))
    def _fill_values(spine_index, time_index, value):
        lookup = {}
        for time, val in zip(time_index, value):
            lookup[time] = float(val)  # because UDF can only return float
        result = []
        for spine_val in spine_index:
            _v = lookup.get(spine_val, interpolate_val)
            result.append(_v)
        return result

    padded_array_df = _fill_values(spine_index, time_index, value)

    return padded_array_df
