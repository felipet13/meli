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
"""API node for creating tags."""
from typing import Dict, List

import pyspark
import pyspark.sql.functions as f
from refit.v1.core.fill_nulls import fill_nulls
from refit.v1.core.has_schema import has_schema
from refit.v1.core.inject import inject_object
from refit.v1.core.input_kwarg_filter import add_input_kwarg_filter
from refit.v1.core.input_kwarg_select import add_input_kwarg_select
from refit.v1.core.output_filter import add_output_filter
from refit.v1.core.retry import retry
from refit.v1.core.unpack import unpack_params

from ...core.tags import generate_tags

# pylint: disable=invalid-name
_tag_col_broadcast = "_tags_all_broadcast"
_tag_col_select = "_tags_all_select"


# pylint: enable=invalid-name
@add_input_kwarg_filter()
@add_input_kwarg_select()
@has_schema()
@unpack_params()
@inject_object()
@retry()
@add_output_filter()
@fill_nulls()
def create_tags_from_config_select(*args, **kwargs) -> pyspark.sql.DataFrame:
    """See core function for more details.

    Add_input_kwarg_filter, add_input_kwarg_select, has_schema, unpack_params,
    inject_object, retry, add_output_filter, fill_nulls
    wrapper for `create_tags_from_config_select`.
    """
    return generate_tags.create_tags_from_config_select(*args, **kwargs)


# pylint: disable=invalid-name
@add_input_kwarg_filter()
@add_input_kwarg_select()
@has_schema()
@unpack_params()
@inject_object()
@retry()
@add_output_filter()
@fill_nulls()
def create_tags_from_config_broadcast(*args, **kwargs) -> pyspark.sql.DataFrame:
    """See core function for more details.

    Add_input_kwarg_filter, add_input_kwarg_select, has_schema, unpack_params,
    inject_object, retry, add_output_filter, fill_nulls
    wrapper for `create_tags_from_config_broadcast`.
    """
    return generate_tags.create_tags_from_config_broadcast(*args, **kwargs)


def create_tags_from_config(
    df: pyspark.sql.DataFrame,
    config: List[Dict[str, str]],
    tag_col_name: str = "tags",
    verbose: bool = False,
) -> pyspark.sql.DataFrame:
    """Calls functions to create tags for a given dataframe and config.

    This is a wrapper function that only works for non-sequential tags:
    1. for tags with input, it calls - `create_tags_from_config_broadcast`
    2. for other tags, it calls - `create_tags_from_config_select`
    Args:
        df: The spark dataframe.
        config: Config for creating tags
        tag_col_name: The name of the tag column name. Defaults to tags.
        verbose: flag used to decide if temp tag columns should be dropped.

    Returns:
        A new spark dataframe with the additional tag column.
    """
    broadcast_tags = []
    select_tags = []

    for tag_param in config:
        if "input" in list(tag_param.keys()):
            broadcast_tags.append(tag_param)
        else:
            select_tags.append(tag_param)

    # get unique columns needed
    list_of_unique_columns = []
    for tag_param in broadcast_tags:
        if tag_param["input"] not in list_of_unique_columns:
            list_of_unique_columns.append(tag_param["input"])

    # sort by input and convert to existing required form
    params_broadcast = {}
    for column in list_of_unique_columns:
        params_broadcast[column] = [
            tag_param for tag_param in broadcast_tags if tag_param["input"] == column
        ]

    df_with_tags_from_broadcast = create_tags_from_config_broadcast(
        df=df,
        config=params_broadcast,
        tag_col_name=_tag_col_broadcast,
    )

    df_with_tags_from_select = create_tags_from_config_select(
        df=df_with_tags_from_broadcast, config=select_tags, tag_col_name=_tag_col_select
    )

    df_with_tags = df_with_tags_from_select.withColumn(
        tag_col_name,
        f.flatten(f.array(_tag_col_broadcast, _tag_col_select)),
    )
    if not verbose:
        df_with_tags = df_with_tags.drop(_tag_col_broadcast, _tag_col_select)

    return df_with_tags
