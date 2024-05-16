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

"""Melt methods."""

from typing import List

import pyspark
from pyspark.sql import functions as f


def _stack_expr(unpivot_col_list: List[str]) -> pyspark.sql.column:
    """Turns a list of columns into a stack expression for ``select`` function.

    Sample input
    ::

    ["uk", "us", "in", "jp"]

    Sample output
    ::
    stack(4, "uk",uk, "us",us, "in",in, "jp",jp)

    Args:
        unpivot_col_list: List of columns to un-pivot

    Returns:
        A stack expression that can be used with ``select``
        or ``withColumn`` function for un-pivot.
    """
    expanded_col_list = ",".join([f"'{x}',{x}" for x in unpivot_col_list])

    n = len(unpivot_col_list)
    stack_expr = f.expr(
        "stack({}, {})".format(  # pylint: disable=consider-using-f-string # noqa: E501
            n, expanded_col_list
        )
    )

    return stack_expr


def melt(
    df: pyspark.sql.DataFrame,
    key_cols: List[str],
    unpivot_col_list: List[str],
    output_key_column: str = "key",
    output_value_column: str = "value",
) -> pyspark.sql.DataFrame:
    """The ``melt`` function un-pivots the data in a pyspark dataframe.

    The input would be a wide format, and output would be returned as
    narrow format.

    Sample inputs
    ::

        +-------+---+---+---+---+
        |product| uk| us| in| jp|
        +-------+---+---+---+---+
        | orange| 12| 23| 17| 12|
        |  mango|  5|  2| 25|  1|
        +-------+---+---+---+---+

    Function call
    ::

        melt(
            df,
            key_cols=["product"],
            unpivot_col_list=["uk", "us", "in", "jp"],
            output_key_column="country",
            output_value_column="quantity",
        )

    Sample Output
    ::

        +-------+-------+--------+
        |product|country|quantity|
        +-------+-------+--------+
        | orange|     uk|      12|
        | orange|     us|      23|
        | orange|     in|      17|
        | orange|     jp|      12|
        |  mango|     uk|       5|
        |  mango|     us|       2|
        |  mango|     in|      25|
        |  mango|     jp|       1|
        +-------+-------+--------+


    Args:
        df: A dataframe in wide format
        key_cols: List of key columns
        unpivot_col_list: List of columns to be un-pivoted
        output_key_column: column name for "key" column
        output_value_column: column name for "value" column

    Returns:
        Returns melted pyspark dataframe.
    """
    stack_expr = _stack_expr(unpivot_col_list)
    out_df = df.select(
        *key_cols, stack_expr.alias(output_key_column, output_value_column)
    )

    return out_df
