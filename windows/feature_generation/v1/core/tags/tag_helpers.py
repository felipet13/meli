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

"""Core utility functions that operate on tag columns."""

from typing import List, Union

import pyspark
import pyspark.sql.functions as f

from ...core.tags.tags import contains_tags


def days_since_first(
    tags: List[str],
    time_col: str,
    windows_spec: Union[pyspark.sql.WindowSpec, List[pyspark.sql.WindowSpec]],
    tag_col: str = "tags",
) -> List[pyspark.sql.Column]:
    """Returns the number of days since the first ``tag`` occurrence.

    It checks if the ``tag`` is there in the ``tag_col`` array. If present,
    it then returns days difference between first ``tag`` and current record.

    Args:
        tags: list of tag names for time since
        time_col: time column to be used for date difference
        windows_spec: PySpark `WindowSpec` objects.
        tag_col: (Optional) column name for tag array

    Returns:
         List of pyspark column with alias that can be used with ``select``.
    """
    if isinstance(windows_spec, pyspark.sql.WindowSpec):
        windows_spec = [windows_spec]

    time_since_cols = []
    for tag in tags:
        tag_timestamp = _if_tag_exists_get_timestamp(
            tag=tag, tag_col=tag_col, time_col=time_col
        )
        for window in windows_spec:
            days_since = f.first(tag_timestamp, ignorenulls=True).over(window)
            time_since_cols.append(
                f.datediff(time_col, days_since).alias(f"days_since_first_{tag}")
            )

    return time_since_cols


def days_since_last(
    tags: List[str],
    time_col: str,
    windows_spec: Union[pyspark.sql.WindowSpec, List[pyspark.sql.WindowSpec]],
    tag_col: str = "tags",
) -> List[pyspark.sql.Column]:
    """Returns the number of days since the last ``tag`` occurrence.

    It checks if the ``tag`` is there in the ``tag_col`` array. If present,
    it then returns days difference between last ``tag`` and current record.

    Args:
        tags: list of tag names for time since
        time_col: time column to be used for date difference
        windows_spec: PySpark `WindowSpec` objects.
        tag_col: (Optional) column name for tag array
    Returns:
         List of pyspark column with alias that can be used with ``select``.
    """
    if isinstance(windows_spec, pyspark.sql.WindowSpec):
        windows_spec = [windows_spec]

    time_since_cols = []
    for tag in tags:
        tag_timestamp = _if_tag_exists_get_timestamp(
            tag=tag, tag_col=tag_col, time_col=time_col
        )
        for window in windows_spec:
            days_since = f.last(tag_timestamp, ignorenulls=True).over(window)
            time_since_cols.append(
                f.datediff(time_col, days_since).alias(f"days_since_last_{tag}")
            )

    return time_since_cols


def _if_tag_exists_get_timestamp(tag, tag_col, time_col) -> pyspark.sql.column:
    """If tag exists, return the timestamp of the tag."""
    return f.when(
        contains_tags(tags=tag, tag_col_name=tag_col), f.col(time_col)
    ).otherwise(f.lit(None))
