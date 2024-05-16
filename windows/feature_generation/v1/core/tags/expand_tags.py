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
"""Core function to expand tags."""
import logging
from typing import Any, Dict, List, Optional

import pyspark
import pyspark.sql.functions as f
from refit.v1.core.make_list_regexable import make_list_regexable

from ...core.aggregation.aggregate import aggregate_attributes
from ...core.tags.tags import (
    convert_tag_to_column,  # noqa: E501
    extract_all_tag_names,
    extract_tag_value,
)

# pylint: disable=line-too-long

logger = logging.getLogger(__name__)


# pylint: disable=unexpected-keyword-arg
@make_list_regexable(
    source_df="df_with_tags",
    make_regexable="params_keep_cols",
    raise_exc=True,
)
def expand_tags(
    df_with_tags: pyspark.sql.DataFrame,
    tags_to_convert: List[str],
    tag_col_name: str = "tags",
    params_keep_cols: Optional[List[str]] = None,
    key_cols: Optional[List[str]] = None,
    column_instructions: Optional[Dict[Any, Any]] = None,
    fillna: Optional[int] = None,
    enable_regex: bool = False,  # pylint: disable=unused-argument
) -> pyspark.sql.DataFrame:
    """Expands tags based on configuration.

    This function expands tags into flags and dynamic tags to column
    values. Output can be a flag(integer type), string type or
    double value column based on whether the tag is dynamic or
    not.

    Args:
        df_with_tags: The aggregated dataframe containing a tag column.
        tags_to_convert: The list of tags to convert.
        tag_col_name: The name of the column containing the array of tags.
        params_keep_cols: The columns to keep from input dataframe. Defaults to None.
        key_cols: List of key columns. (eg: pred_entity_id)               # column representing the expected PK, also accepts a list of string for composite PK  # noqa: E501
        column_instructions: Dictionary with column transitions.
        fillna: Value to replace missing entries
        enable_regex: Allows regex selection within `params_keep_cols`. Defaults to
            False.

    Returns:
        A new wide pyspark dataframe where tags have been converted to flags.
    """
    keep_cols = params_keep_cols if params_keep_cols else ["*"]

    df_with_tag_columns = df_with_tags.select(
        *keep_cols,
        *[
            extract_tag_value(
                convert_tag_to_column(tag=x, tag_col_name=tag_col_name), alias=x
            )
            for x in tags_to_convert
        ],
    )

    if column_instructions and key_cols:
        df_with_tags_extracted = aggregate_attributes(
            df=df_with_tag_columns,
            key_cols=key_cols,
            column_instructions=column_instructions,
        )
        subset_cols = list(set(df_with_tags_extracted.columns) - set(key_cols))

    else:
        df_with_tags_extracted = df_with_tag_columns
        subset_cols = tags_to_convert

    if fillna is not None:
        return df_with_tags_extracted.fillna(fillna, subset=subset_cols)

    return df_with_tags_extracted


def _get_tag_list(df_with_tags: pyspark.sql.DataFrame, tag_col_name: str) -> List[str]:
    """Returns the list of tags present in the dataframe.

    This function checks all tags present in `tag_col_name` column and retyrns
    a list of all the tags.

    Args:
        df_with_tags: The aggregated dataframe containing a tag column.
        tag_col_name: The name of the column containing the array of tags.

    Returns:
        A list of all the tags present in df_with_tags.
    """
    df = (
        df_with_tags.withColumn("tags", extract_all_tag_names(tag_col_name))
        .withColumn("feature_name", f.explode_outer(f.col("tags")))
        .select("feature_name")
        .distinct()
        .dropna()
    )
    return [row.feature_name for row in df.collect()]


def expand_tags_all(
    df_with_tags: pyspark.sql.DataFrame,
    tag_col_name: str = "tags",
    params_keep_cols: Optional[List[str]] = None,
    key_cols: Optional[List[str]] = None,
    column_instructions: Optional[Dict[Any, Any]] = None,
    fillna: Optional[int] = None,
    enable_regex: bool = False,  # pylint: disable=unused-argument
) -> pyspark.sql.DataFrame:
    """Thin wrapper to run expand tags but with all tags expanded.

    The function first calls `_get_tag_list` to get the list of tags in the input
    dataframe. Then it calls the `expand_tags` with this list.

    Args:
        df_with_tags: The aggregated dataframe containing a tag column.
        tag_col_name: The name of the column containing the array of tags.
        params_keep_cols: The columns to keep from input dataframe. Defaults to None.
        key_cols: List of key columns. (eg: pred_entity_id)               # column representing the expected PK, also accepts a list of string for composite PK  # noqa: E501
        column_instructions: Dictionary with column transitions.
        fillna: Value to replace missing entries
        enable_regex: Allows regex selection within `params_keep_cols`. Defaults to
            False.

    Returns:
        A new wide pyspark dataframe where tags have been converted to flags.
    """
    tag_list = _get_tag_list(df_with_tags, tag_col_name)

    return expand_tags(
        df_with_tags=df_with_tags,
        tags_to_convert=tag_list,
        tag_col_name=tag_col_name,
        params_keep_cols=params_keep_cols,
        key_cols=key_cols,
        column_instructions=column_instructions,
        fillna=fillna,
    )


# pylint: disable=too-many-arguments


def expand_tags_with_spine(
    df_with_tags: pyspark.sql.DataFrame,
    df_spine: pyspark.sql.DataFrame,
    spine_cols: List[str],
    tags_to_convert: List[str],
    tag_col_name: str = "tags",
    params_keep_cols: Optional[str] = None,
    key_cols: Optional[List[str]] = None,
    column_instructions: Optional[Dict[Any, Any]] = None,
    fillna: Optional[int] = None,
) -> pyspark.sql.DataFrame:
    """Thin wrapper to run expand tags but join to spine after.

    This function exists to reduce amount of pipelining work.
    """
    df_before_spine = expand_tags(
        df_with_tags=df_with_tags,
        tags_to_convert=tags_to_convert,
        tag_col_name=tag_col_name,
        params_keep_cols=params_keep_cols,
        key_cols=key_cols,
        column_instructions=column_instructions,
        fillna=fillna,
    )

    return df_spine.join(df_before_spine, on=spine_cols, how="left")
