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

"""Imputation using forward or backward filling."""

from typing import List, Union

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Window

from ..utils.alias import alias


@alias()
def forward_fill(
    column: pyspark.sql.Column,
    partition_by: Union[str, List[str]],
    order_by: Union[str, List[str]],
) -> pyspark.sql.Column:
    """Performs forward filling on a column given window specifications.

    Use `alias` argument for setting the name of the output column.

    ::

        +----------+------+-------+
        |date      |y_flag|ffilled|
        +----------+------+-------+
        |2012-05-06|0     |0      |
        |2012-05-07|null  |0      |
        |2012-05-08|1     |1      |
        |2012-05-09|null  |1      |
        |2012-05-10|2     |2      |
        +----------+------+-------+

    Args:
        column: The name of the column.
        partition_by: The column(s) to partition by.
        order_by: The column(s) to order by.

    Returns:
        A forward filled column.
    """
    if isinstance(partition_by, str):
        partition_by = [partition_by]

    if isinstance(order_by, str):
        order_by = [order_by]

    ffilled = f.last(f.col(column), ignorenulls=True).over(
        Window.partitionBy(*partition_by).orderBy(*order_by)
    )

    return f.coalesce(f.col(column), ffilled)


@alias()
def backward_fill(
    column: pyspark.sql.Column,
    partition_by: Union[str, List[str]],
    order_by: Union[str, List[str]],
) -> pyspark.sql.Column:
    """Performs backward filling on a column given window specifications.

    Use `alias` argument for setting the name of the output column.

    ::

        +----------+------+-------+
        |date      |y_flag|bfilled|
        +----------+------+-------+
        |2012-05-06|0     |0      |
        |2012-05-07|null  |1      |
        |2012-05-08|1     |1      |
        |2012-05-09|null  |2      |
        |2012-05-10|2     |2      |
        +----------+------+-------+

    Args:
        column: The name of the column.
        partition_by: The column(s) to partition by.
        order_by: The column(s) to order by.

    Returns:
        A backward filled column.
    """
    if isinstance(partition_by, str):
        partition_by = [partition_by]

    if isinstance(order_by, str):
        order_by = [order_by]

    order_by = [f.col(x).desc() for x in order_by]

    backfilled = f.last(f.col(column), ignorenulls=True).over(
        Window.partitionBy(*partition_by).orderBy(*order_by)
    )

    return f.coalesce(f.col(column), backfilled)
