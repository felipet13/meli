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
"""Experimental: Module for dynamic windows."""

import pyspark
import pyspark.sql.functions as f

from ..utils.alias import alias


@alias()
def dynamic_window(
    array: str, lower_bound: str, upper_bound: str, anchor: str
) -> pyspark.sql.Column:
    """Returns an array based on window specs defined within a column.

    Use `alias` argument for setting the name of the output column.

    Prerequisite is that an unbounded preceding and unbounded following ordered window
    be used to collect the array. Either -

    ::
    # method 1 - Collect the entire window using select
    w = (
    Window.partitionBy("person")
    .orderBy("time_index")
    .rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )
    df = df.withColumn(
    "entire_window", f.collect_list(f.struct("time_index", "value")).over(w)
    )

    # method 2 - Collect the entire window using groupby using pure pyspark function
    df.groupBy("person").agg(
    f.array_sort(f.collect_list(f.struct("time_index", "value"))).alias("ordered")
    )

    # method 3 - Use module function
    df.groupBy("person").agg(
    ordered_collect_list("time_index", "value").alias("ordered_collect_list")
    )


    Args:
        array: The array column to apply the dynamic window on.
        lower_bound: The lower bound for the window, an integer column.
        upper_bound: The upper bound for the window, an integer column.
        anchor: The reference point for the upper and lower bounds, an integer column.

    Returns:
        A new pyspark array column filtered by the window specs.
    """
    filtered_window_expr = f.expr(
        f"""
        filter(
            {array},
            x -> (
                ({anchor}+{lower_bound}) <= x.{anchor})
                AND
                (x.{anchor} <= ({anchor}+{upper_bound})
            )
        )"""
    )
    return filtered_window_expr
