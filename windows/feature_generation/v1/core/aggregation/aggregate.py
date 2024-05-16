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
# pylint: disable=line-too-long
"""Core module to create aggregation."""


import logging
from typing import Dict, List

import pyspark


def aggregate_attributes(
    df: pyspark.sql.DataFrame,
    key_cols: List[str],
    column_instructions: Dict[str, List[pyspark.sql.Column]],
) -> pyspark.sql.DataFrame:
    """Parameter driven attribute selection per column, given an entity mapping table.

    The format for column_instructions is:
    {
    "output_col1": min(col)                                             # defaults to pyspark.sql.functions if arg is function  # noqa: E501
    "output_col2": my_udf_wrapper(collect_list(col2))                   # will take the input column to pass to udf if arg is udf  # noqa: E501
    "output_col3": udf_wrapper(collect_list(col1), collect_list(col2))  # will take multiple input columns and pass to udf in that order  # noqa: E501
    "output_col5": func(col1, arg1: argument1, arg2: argument2)         # pass keyword arguments where necessary # noqa: E501
    }

    Args:
        df: The entity mapping table with attributes.
        key_cols: List of key columns. (eg: pred_entity_id)               # column representing the expected PK, also accepts a list of string for composite PK  # noqa: E501
        column_instructions: Dictionary with column transitions.

    Returns:
        A new pyspark dataframe with id_col, and all other attributes have been reduced
        to single row.

    Raises:
        AttributeError: A vaild function object should be passed.
    """
    columns_to_aggregate = []
    for new_column, func in column_instructions.items():
        try:
            columns_to_aggregate.append(func.alias(new_column))
        except AttributeError as error:  # noqa: PERF203
            logging.error(error)
            logging.error("A valid function object should be passed.")

    new_df = df.groupBy(*key_cols).agg(
        *columns_to_aggregate,
    )
    return new_df
