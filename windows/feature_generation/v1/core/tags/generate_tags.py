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
# noqa: C0302
"""Core module to create tags.

Spark and parquet files are known
to suffer performance issues when there are too many columns. This sub-module
attempts to take advantage of pyspark's native ability to handle arrays. A user
has to decide when to use tags instead of flags, as flags are easier to handle.
"""

import logging
from typing import Any, Dict, List, Mapping

import pyspark
import pyspark.sql.functions as f

from ...core.tags.tags import CONSTANTS, construct_tag

_PREFIX = "_tag_"
_NULL = "_null"

logger = logging.getLogger(__name__)


def create_tags_from_config_select(
    df: pyspark.sql.DataFrame,
    config: List[Mapping[str, Any]],
    tag_col_name: str = "tags",
    sequential: bool = False,
) -> pyspark.sql.DataFrame:
    """Creates tags given a dataframe and a tag config.

    Defaults function default module location.

    Args:
        df: The spark dataframe.
        config: The tag config.
        tag_col_name: The name of the tag column name. Defaults to tags.
        sequential: Whether to run the flag generation sequentially. This allows
        flags to depend on previously created flags. Defaults to False.

    Returns:
        A new spark dataframe with the additional tag column.
    """
    if tag_col_name not in df.columns:
        df = df.withColumn(tag_col_name, f.array(construct_tag(tag=_NULL)))

    new_tag_col = []

    for element in config:
        if isinstance(element, list):
            new_tag_col.extend(element)
        else:
            new_tag_col.append(element)

    if not sequential:
        df_with_tags = df.select("*", *new_tag_col)
        df_with_tags = _post_process_tags_df(
            df_with_tags=df_with_tags, tag_col_name=tag_col_name
        )

        return df_with_tags

    logger.info(
        "Creating tags sequentially. Use .checkpoint() if you are "
        "hitting a stackoverflow error (hint: google 'spark checkpoint df'). "
        "This happens because the spark DAG is too long. Alternatively, you "
        "can break up the creation of your tags into chunks."
    )
    df_with_tags = df
    for flag in new_tag_col:
        df_with_tags = df_with_tags.select("*", flag)
        df_with_tags = _post_process_tags_df(
            df_with_tags=df_with_tags, tag_col_name=tag_col_name
        )

    return df_with_tags


def _post_process_tags_df(
    df_with_tags: pyspark.sql.DataFrame, tag_col_name: str
) -> pyspark.sql.DataFrame:
    """Post processing of tags after generating them.

    Each tag is generated as a column, but subsequently flattened into a single
    array column.
    """
    tag_cols = [x for x in df_with_tags.columns if x.startswith(_PREFIX)]

    df_with_tags = (
        df_with_tags.withColumn(
            tag_col_name,
            f.flatten(f.array(*tag_cols, f.col(tag_col_name))),
        )
        .withColumn(
            tag_col_name,
            f.expr(f"filter({tag_col_name}, x -> x.{CONSTANTS.TAG} != '{_NULL}')"),
        )
        .drop(*tag_cols)
    )
    return df_with_tags


# pylint: disable=invalid-name
def create_tags_from_config_broadcast(  # noqa: C901
    df: pyspark.sql.DataFrame,
    config: Dict[str, List[pyspark.sql.Column]],
    tag_col_name: str = "tags",
) -> pyspark.sql.DataFrame:
    """Creates tags given a dataframe and a tag config using broadcasting.

    This has been shown to be much more performant on large data, but the trade-off is
    that it only works for functions with the `input` argument.

    Args:
        df: The spark dataframe.
        config: The tag config.
        tag_col_name: The name of the tag column name. Defaults to tags.

    Returns:
        A new spark dataframe with the additional tag column.

    Raises:
        ValueError: If input not present in tag config.
    """
    # get unique columns needed
    list_of_unique_columns = list(config.keys())

    # collect unique values within each column and put into their own dataframe
    dict_of_dfs_with_unique_values_only = {}
    for column in list_of_unique_columns:
        dict_of_dfs_with_unique_values_only[column] = df.select(column).dropDuplicates()

    # run actual tagging on unique values dfs
    dict_of_dfs_with_tags_separate = {}
    new_columns = []
    for column in list_of_unique_columns:
        dict_of_dfs_with_tags_separate[column] = create_tags_from_config_select(
            df=dict_of_dfs_with_unique_values_only[column],
            config=config[column],
            tag_col_name="only_" + column,
        )
        new_columns.append("only_" + column)

    # bring back tag columns to original df
    modified_df = df
    original_columns_order = df.columns

    for column, _df in dict_of_dfs_with_tags_separate.items():
        modified_df = modified_df.join(f.broadcast(_df), on=[column], how="left")
        modified_df = modified_df.withColumn(
            "only_" + column,  # post processing of tag columns
            f.when(
                (f.size(f.col("only_" + column)) == f.lit(0))
                | f.col("only_" + column).isNull(),
                f.array(
                    f.struct(
                        f.lit(_NULL).alias(CONSTANTS.TAG),
                        f.lit(1).alias(CONSTANTS.VALUE),
                    )
                ),
            ).otherwise(f.col("only_" + column)),
        )

    if new_columns:
        df_with_tags = (
            modified_df.select(
                *original_columns_order,
                f.flatten(f.array(*[f.col(x) for x in new_columns])).alias(
                    tag_col_name
                ),
            )
            .withColumn(
                tag_col_name,
                f.expr(f"filter({tag_col_name}, x -> x.{CONSTANTS.TAG} != '{_NULL}')"),
            )
            .drop(*new_columns)
        )
    else:
        modified_df = modified_df.withColumn(
            tag_col_name, f.array(construct_tag(tag=_NULL))
        )
        df_with_tags = modified_df.withColumn(
            tag_col_name,
            f.expr(f"filter({tag_col_name}, x -> x.{CONSTANTS.TAG} != '{_NULL}')"),
        ).drop(*new_columns)

    return df_with_tags
