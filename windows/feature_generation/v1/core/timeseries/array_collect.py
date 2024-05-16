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
# pylint: disable=invalid-name
"""Module to collect columns into arrays."""

from typing import Callable, List, Optional, Union

import pyspark
import pyspark.sql.functions as f

from ..utils.alias import alias


def collect_array_then_interpolate(
    df: pyspark.sql.DataFrame,
    order: str,
    values: Union[str, List[str]],
    groupby: Union[str, List[str]],
    interpolate_func: Callable,
    desc: bool = False,
    spine: Optional[str] = None,
    delta: int = 1,
    **interpolate_func_kwargs,
) -> pyspark.sql.DataFrame:
    """Collects the `value` and `order` column into an array then pads them out.

    It also generates a spine array column with an array of scalar values.
    The scalar values represent the timestamp in epoch seconds.
    Additionally, interpolates the missing `value` in a separate column.

    Args:
        df: The pyspark dataframe.
        order: The name of the column that defines the order to sort the values array.
            Should be an index.
        values: The name of the columns with the values to collect.
        groupby: The name of the column or columns that define the unit of aggregation.
        interpolate_func: function to be used for interpolation.
            Supported functions are interpolate_constant and scipy_interpolate.
        desc: Defines if values should be collected in descending order.
        spine: The column name for the spine. If not specified a spine will not be
            generated.
        delta: The interval to generate a spine. Should be in the same units as the
            defined index for `order` column.
        interpolate_func_kwargs: kwargs for interpolate function.

    Returns:
        A new pyspark dataframe with the `value` columns collected to arrays.
    """
    if isinstance(groupby, str):
        groupby = [groupby]

    if isinstance(values, str):
        values = [values]

    collected_df = (
        df.groupBy(*groupby).agg(
            sorted_collect_list(order=order, values=values, desc=desc).alias(
                "_array_of_struct"
            )
        )
    ).select(
        *groupby,
        *[f.col(f"_array_of_struct.{val}").alias(val) for val in values],
        f.col(f"_array_of_struct.{order}").alias(order),
    )

    spine = spine or "spine_index"
    collected_df_with_spine = create_spine_array_from_index_array(
        df=collected_df, order_array=order, delta=delta, alias=spine
    )
    result_df = collected_df_with_spine.select(
        "*",
        *[
            interpolate_func(
                spine_index=spine,
                time_index=order,
                value=x,
                alias=f"{x}_array_padded",
                **interpolate_func_kwargs,
            )
            for x in values
        ],
    )
    return result_df


# pylint: disable=invalid-name,redefined-outer-name
def create_spine_array_from_index_array(
    df: pyspark.sql.DataFrame, order_array: str, alias: str, delta: int = 1
) -> pyspark.sql.DataFrame:
    """Create spine array using index array.

    Args:
        df: The pyspark dataframe.
        order_array: The name of the array column that contains the index
            representing order.
        delta: The interval to generate a spine. Should be in the same units as the
            defined index for `order` column.
        alias: Output alias.

    Returns:
        Pyspark dataframe with spine array column.
    """
    result = (
        df.select(
            "*",
            f.array_max(order_array).alias("_max_index"),
            f.array_min(order_array).alias("_min_index"),
        )
        .select(
            "*",
            f.sequence(start="_min_index", stop="_max_index", step=f.lit(delta)).alias(
                alias
            ),
        )
        .drop("_max_index", "_min_index")
    )

    return result


@alias()
def sorted_collect_list(
    order: str,
    values: List[str],
    desc: bool = False,
) -> pyspark.sql.Column:
    """Collect a list in the order specified.

    Use `alias` argument for setting the name of the output column.

    Args:
        order: The column used to determine the window order.
        values: The value to collect.
        desc: Whether to reverse the sorting of the array.

    Returns:
        A pyspark array of struct column.
    """
    if isinstance(values, str):
        values = [values]

    sorted_array = f.array_sort(f.collect_list(f.struct(order, *values)))

    if desc:
        return f.reverse(sorted_array)
    return sorted_array
