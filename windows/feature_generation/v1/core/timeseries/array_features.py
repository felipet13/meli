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
"""Computes array features."""

from typing import Callable, List

import pyspark


def create_array_features(
    df: pyspark.sql.DataFrame,
    columns: List[Callable],
) -> pyspark.sql.DataFrame:
    """Creates aggregates or transformations from array columns sequentially.

    It can leverage any function in `array_transform.py` and `array_aggregate.py`.

    Args:
        df: The pyspark dataframe.
        columns: Instructions to compute features from array columns.

    Returns:
        A new pyspark dataframe with the `value` columns collected to arrays.
    """
    result_df = df
    for callable_to_apply in columns:
        result_df = result_df.select("*", callable_to_apply)

    return result_df
