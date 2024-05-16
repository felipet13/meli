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

"""Internal helper schema functions."""

from typing import List

import pyspark
from pyspark.sql import functions as f


def _get_full_schema(list_of_dataframes: List[pyspark.sql.DataFrame]) -> List[str]:
    """Return superset of all columns in all dataframes."""
    list_of_columns = []
    for dataframe in list_of_dataframes:
        for col in dataframe.columns:
            if col not in list_of_columns:
                list_of_columns.append(col)
    return list_of_columns


def _add_missing_cols(
    list_of_columns: List[str], df: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """Adds missing columns to dataframe as ``nulls``."""
    missing_cols = set(list_of_columns) - set(df.columns)
    modified_df = df.select("*", *[f.lit(None).alias(x) for x in missing_cols]).select(
        *list_of_columns
    )
    return modified_df
