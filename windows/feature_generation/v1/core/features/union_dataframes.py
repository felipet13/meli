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
"""Union tables together."""

import logging
from typing import Optional

import pyspark

from ...core.utils.union import reduce_union

logger = logging.getLogger(__name__)


def reduce_union_dataframes(
    union_by_name: bool = False,
    allow_missing_columns: Optional[bool] = False,
    **dfs: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """Union any number of dataframes.

    Different function signature to conform to kedro's node API.

    Args:
        union_by_name: Whether to union dataframe by column names. Defaults to False.
        allow_missing_columns: Whether to allow missing columns. Used only for the
            union_by_name case. Defaults to False.
        **dfs: dataframes to union.

    Returns:
        Merged dataframe.
    """
    list_of_dataframes = list(dfs.values())

    return reduce_union(
        list_of_dataframes=list_of_dataframes,
        union_by_name=union_by_name,
        allow_missing_columns=allow_missing_columns,
    )
