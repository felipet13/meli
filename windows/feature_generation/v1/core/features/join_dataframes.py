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
"""Joins tables together."""

import logging
from typing import Optional

import pyspark

logger = logging.getLogger(__name__)


def join_dataframes_with_spine(
    spine_df: pyspark.sql.DataFrame,
    default_join_keys: list,
    alternative_join_mapping: Optional[dict] = None,
    **dfs: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """Creates table by joining feature/target dataframes to spine.

    Each dataframe is joined on the column of the spine. Alternatively other
    combinations for joining keys can be defined in the instructions.

    Args:
        spine_df: spine
        default_join_keys: default join keys
        alternative_join_mapping: join keys and join type for dataframes not following
            default. Defaults to None.
        **dfs: feature/target dataframes to join

    Returns:
        The merged spark dataframe.
    """
    final_df = spine_df
    alternative_join_mapping = alternative_join_mapping or {}

    for df_name, df in dfs.items():
        if alternative_join_mapping.get(df_name):
            how = alternative_join_mapping[df_name].get("join_type", "left")
            join_keys = alternative_join_mapping[df_name].get(
                "join_keys", default_join_keys
            )
        else:
            how = "left"
            join_keys = default_join_keys
        final_df = final_df.join(df, on=join_keys, how=how)

    return final_df
