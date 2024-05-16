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

"""Column level aggregation metrics to derive features."""

from typing import Callable, List, Optional, Union

import pyspark
from pyspark.sql import Window
from pyspark.sql import functions as f

from ..utils.alias import alias


@alias()
def add_aggregate_value_column(
    partition_by: Union[str, List[str]],
    func: Callable,
) -> pyspark.sql.Column:
    """Adds a column with aggregate value of the value column in the partition.

    A single call returns aggregated values depending on the parameters.
    Use `alias` argument for setting the name of the output column.

    Args:
        partition_by: Column(s) to partition the DataFrame by.
        func: The aggregation type to be applied for value column.

    Returns:
        Column: Returns column with aggregate value on the data
            partitioned by partition_by.
    """
    if isinstance(partition_by, str):
        partition_by = [partition_by]

    window = Window.partitionBy(partition_by)
    return func.over(window)


@alias()
def auc_trapezoidal(
    y: str,
    x: str,
    partition_by: Union[str, List[str]],
) -> pyspark.sql.Column:
    """Calculates area-under-the-curve (AUC) using the trapezoidal method.

    Use `alias` argument for setting the name of the output column.

    Args:
        y: Column which is y-axis of the curve.
        x: Column used to sort y-axis values. Must be of numeric type or timestamp.
        partition_by: Column(s) to partition the DataFrame by when getting lagged
        values for x and y.

    Returns:
        Column: Returns column with area between 2
            consecutive points ordered by order_by column.
    """
    if isinstance(partition_by, str):
        partition_by = [partition_by]

    window = Window.partitionBy(partition_by).orderBy(x)  # noqa: E501

    # Variables used for trapezoidal area calculations
    y_1 = f.lag(y, default=y).over(window)
    y_2 = f.col(y)
    x_1 = f.lag(f.col(x).cast("long"), default=x).over(window)
    x_2 = f.col(x).cast("long")

    area = 0.5 * (y_1 + y_2) * (x_2 - x_1)

    return area


@alias()
def weighted_avg(
    partition_by: Union[str, List[str]],
    y: str,
    x: str,
    is_weight_sequence: bool = True,
) -> pyspark.sql.Column:
    """Calculate weighted avg.

    Use `alias` argument for setting the name of the output column.

    Args:
        partition_by: Logical group of columns that define the granularity
        required to calculate weighted avg on.
        y: Column with numerical values to aggregate using weighted avg.
        x: Column with weights required for the weighted avg.
        is_weight_sequence: Boolean indicating whether the weight column is a sequence.
        When True, weight will be calculated as a difference between current and
        the next value.

    Returns:
        Column providing the average of y weighted by x over the defined partitions
    """
    window = Window.partitionBy(partition_by)
    if is_weight_sequence:
        order_window = Window.partitionBy(partition_by).orderBy(x)
        x_2 = f.lead(f.col(x), default=x).over(order_window).cast("long")
        x_1 = f.col(x).cast("long")
        weights = x_2 - x_1
    else:
        weights = f.col(x)

    weight_value = weights * f.col(y)
    weight_value_sum = f.sum(weight_value).over(window)
    weight_sum = f.sum(weights).over(window)

    wt_avg = weight_value_sum / weight_sum

    return wt_avg


@alias()
def abs_relative_variation(
    val_col: pyspark.sql.Column, ref_col: pyspark.sql.Column, round_dp: int = 2
) -> pyspark.sql.Column:
    """Calculated the absolute variation wtr to a reference column.

    Use `alias` argument for setting the name of the output column.

    Args:
        val_col: value column
        ref_col: reference value column
        round_dp: decimal places to round the calculated variation

    Returns:
        The absolute variation between a value and a reference value
    """
    return f.bround(f.abs(1 - val_col / ref_col), round_dp)


@alias()
def find_local_peaks(  # pylint: disable=too-many-locals
    partition_by: Union[str, List[str]],
    order_by: str,
    value_col: str,
    range_window: int,
    tol_val: Optional[float],
) -> pyspark.sql.Column:
    """Finds local peaks and their ranks.

     The function creates a rolling average window for the given range.
     It then uses this rolling average to determine the local peaks and their ranks.

     Use `alias` argument for setting the name of the output column.

    Args:
        partition_by: Logical group of columns that define the granularity
        required to calculate local peaks on.
        order_by: Column to order the data when looking for local peaks
        (timestamp column casted  as `long` datatype).
        value_col: Column used to find peaks.
        range_window: Range to calculate rolling averages (in minutes).
        tol_val: Value for accepted tolerance when comparing tag values.

    Returns:
        Column providing the local peaks(minima, maxima
        and their respective ranks separated by '_').
    """
    if isinstance(partition_by, str):
        partition_by = [partition_by]
    win = (
        Window.partitionBy(partition_by)
        .orderBy(f.col(order_by))
        .rangeBetween(-(range_window * 60), 0)
    )
    lag_win = Window.partitionBy(partition_by).orderBy(f.col(order_by))

    rol_avg = f.avg(value_col).over(win)
    rol_avg_lag = f.lag(rol_avg).over(lag_win)
    rol_avg_lead = f.lead(rol_avg).over(lag_win)
    if tol_val:
        rol_avg_tol = rol_avg + tol_val
        rol_avg_lag_tol = rol_avg_lag + tol_val
        rol_avg_lead_tol = rol_avg_lead + tol_val
    else:
        rol_avg_tol = rol_avg
        rol_avg_lag_tol = rol_avg_lag
        rol_avg_lead_tol = rol_avg_lead
    is_peak = (
        f.when(
            (
                ((rol_avg >= rol_avg_lag_tol) & (rol_avg > rol_avg_lead_tol))
                | ((rol_avg > rol_avg_lag_tol) & (rol_avg >= rol_avg_lead_tol))
            ),
            "maxima",
        )
        .when(
            (
                ((rol_avg_tol <= rol_avg_lag) & (rol_avg_tol < rol_avg_lead))
                | ((rol_avg_tol < rol_avg_lag) & (rol_avg_tol <= rol_avg_lead))
            ),
            "minima",
        )
        .otherwise(None)
    )
    partition_cols = [*partition_by, is_peak]
    peak_rank = f.dense_rank().over(
        Window.partitionBy(partition_cols).orderBy(f.asc(order_by))
    )

    return f.concat(is_peak, f.lit("_"), peak_rank)
