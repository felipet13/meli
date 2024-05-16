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
"""Creates interaction features from numerical columns."""

import itertools
import logging
import re
from functools import reduce
from operator import mul
from typing import Dict, List, Optional

import pyspark
import pyspark.sql.functions as f

logger = logging.getLogger(__name__)


def create_interaction_features(
    df: pyspark.sql.DataFrame,
    params_interaction: List[Dict[str, List[str]]],
    params_spine_cols: Optional[List[str]] = None,
    keep_new_cols: bool = True,
) -> pyspark.sql.DataFrame:
    """Creates interaction features given a list of interaction dictionaries.

    Args:
        df: A pyspark dataframe.
        params_interaction: A dictionary containing a list of regex patterns to create
            interaction features.
        params_spine_cols: Spine columns.
        keep_new_cols: Whether to keep only the new columns and spine columns.
            Defaults to True.

    Returns:
        A new pyspark dataframe with additional interaction features as columns.
    """  # noqa: E501
    column_combination_list = []
    df_columns = df.columns
    for instructions in params_interaction:
        columns_dict = {}
        for key, list_of_regexes in instructions.items():
            matched_cols = _extract_elements_in_list(df_columns, list_of_regexes)
            if matched_cols:
                columns_dict[key] = matched_cols
        column_combination_list.append(columns_dict)

    logger.info("columns_dict: %s", columns_dict)

    all_combinations = []
    for columns_dict in column_combination_list:
        all_combinations.extend(list(itertools.product(*list(columns_dict.values()))))

    new_df = df.select(
        "*", *[_multiply_all_elements(combi) for combi in all_combinations]
    )

    new_columns = _difference_in_columns(new_df, df)
    # do something with new columns, like built in groupby or whatever
    logger.info("New interaction features generated: %s", new_columns)

    if keep_new_cols:
        return new_df.select(*params_spine_cols, *new_columns)

    return new_df


def _extract_elements_in_list(
    full_list_of_columns: List[str], list_of_regexes: List[str]
) -> List[str]:
    """Use regex to extract elements in a list."""
    results = []
    for regex in list_of_regexes:
        matches = list(filter(re.compile(regex).match, full_list_of_columns))
        if matches:
            for match in matches:
                if match not in results:
                    results.append(  # helps keep relative ordering as defined in YAML
                        match
                    )
        else:
            logger.warning("The following regex did not return a result: %s", regex)
    return results


def _multiply_all_elements(list_of_strings: List[str]) -> pyspark.sql.Column:
    """Constructs a pyspark multiplication for all elements in a list."""
    list_of_cols = [f.col(x) for x in list_of_strings]
    new_col_name = "_".join(list_of_strings)
    formula = reduce(mul, list_of_cols)
    return formula.alias(new_col_name)


def _difference_in_columns(
    new_df: pyspark.sql.DataFrame, old_df: pyspark.sql.DataFrame
) -> List[str]:
    """Given two dataframes, finds the difference in columns.

    Note: Retains relative ordering of columns for aesthetic reasons.
    """
    new_columns = new_df.columns
    old_columns = old_df.columns

    difference_in_columns = [
        new_col for new_col in new_columns if new_col not in old_columns
    ]
    return difference_in_columns
