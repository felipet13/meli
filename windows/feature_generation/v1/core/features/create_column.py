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
"""Create column function."""

from typing import Callable, List, Optional

from pyspark.sql import Column, DataFrame
from refit.v1.core.make_list_regexable import make_list_regexable


@make_list_regexable(
    source_df="df",
    make_regexable="params_keep_cols",
    raise_exc=True,
)
def create_columns_from_config(
    df: DataFrame,
    column_instructions: List[Callable],
    sequential: bool = False,
    params_keep_cols: Optional[List[str]] = None,
    enable_regex: bool = False,  # pylint: disable=unused-argument
) -> DataFrame:
    """Creates new columns on a dataframe according to a config file.

    Args:
        df: DataFrame to search using yaml instructions
        column_instructions:a dictionary with the new column name and function to apply.
        All new columns will be part of the output dataframe.
        sequential: Whether to run the flag generation sequentially. This allows
        flags to depend on previously created flags. Defaults to False.
        params_keep_cols: The columns to keep from input dataframe. Defaults to None.
        enable_regex: Allows regex selection within params_keep_cols. Defaults to
        False.

    Returns:
        A spark dataframe with additional columns.
    """
    params_keep_cols = params_keep_cols or df.columns

    new_columns = []
    for column_instruction in column_instructions:
        new_column = column_instruction
        if isinstance(new_column, Column):
            new_column = [new_column]

        new_columns.extend(new_column)

    if not sequential:
        return df.select(*params_keep_cols, *new_columns)

    drop_cols = list(set(df.columns) - set(params_keep_cols))
    for column in new_columns:
        df = df.select("*", column)

    return df.drop(*drop_cols)
