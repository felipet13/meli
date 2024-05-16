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

"""Union methods rewritten as functions."""

from functools import reduce
from typing import List

import pyspark

from ...core.utils.schema import _add_missing_cols, _get_full_schema


def reduce_union(
    list_of_dataframes: List[pyspark.sql.DataFrame],
    union_by_name: bool = False,
    allow_missing_columns: bool = False,
) -> pyspark.sql.DataFrame:
    """Given a list of dataframes, union all of them into a single dataframe.

    Args:
        list_of_dataframes: A list of dataframes with equivalent schemas.
        union_by_name: Whether to union dataframe by column names. Defaults to False.
        allow_missing_columns: Whether to allow missing columns. Used only for the
            union_by_name case. Defaults to False.

    Returns:
        A spark dataframe.
    """
    if union_by_name:
        return reduce(
            lambda x, y: x.unionByName(y, allowMissingColumns=allow_missing_columns),
            list_of_dataframes,
        )

    return reduce(lambda x, y: x.union(y), list_of_dataframes)


def lazy_union(
    list_of_dataframes: List[pyspark.sql.DataFrame],
) -> pyspark.sql.DataFrame:
    """Unions a list of dataframes that may not have consistent schema.

    Note: Order of output columns is determined by the ordering of the columns in the
        dataframes in the list, i.e. the earlier the column appears in the list, the
        more likely it will be on the left hand side of the output dataframe.

    Args:
        list_of_dataframes: The list of dataframes to union.

    Returns:
        A spark dataframe.
    """
    list_of_columns = _get_full_schema(list_of_dataframes)

    modified_list_of_dataframes = []
    for dataframe in list_of_dataframes:
        modified_df = _add_missing_cols(list_of_columns, dataframe)
        modified_list_of_dataframes.append(modified_df)

    return reduce_union(modified_list_of_dataframes)
