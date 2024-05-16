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

"""Contains an upsert function."""

from typing import List, Union

import pyspark
from pyspark.sql import functions as f

from ...core.utils.schema import _add_missing_cols, _get_full_schema


def upsert(
    df_old: pyspark.sql.DataFrame,
    df_new: pyspark.sql.DataFrame,
    join_key: Union[str, List[str]],
) -> pyspark.sql.DataFrame:
    """Upserts given 2 tables where the new dataframe takes precedence.

    Also handles uneven schemas for the tables where new columns are added but old
    columns are updated.

    Args:
        df_old: The older dataframe.
        df_new: The new dataframe with updated attributes.
        join_key: The join key of the 2 tables.

    Returns:
        A new pyspark dataframe.
    """
    if isinstance(join_key, str):
        join_key = [join_key]

    full_schema = _get_full_schema([df_old, df_new])
    df_old = _add_missing_cols(full_schema, df_old)
    df_new = _add_missing_cols(full_schema, df_new)

    attr_cols = [x for x in df_old.columns if x not in join_key]

    _left_attr_cols_name = ["l_" + x for x in attr_cols]
    _right_attr_cols_name = ["r_" + x for x in attr_cols]

    _left_attr_cols = [
        f.col(x).alias(y) for x, y in zip(attr_cols, _left_attr_cols_name)
    ]
    _right_attr_cols = [
        f.col(x).alias(y) for x, y in zip(attr_cols, _right_attr_cols_name)
    ]

    _df_old = df_old.select(*join_key, *_left_attr_cols)
    _df_new = df_new.select(*join_key, *_right_attr_cols)

    _coalesce = [
        f.coalesce(y, x).alias(z)
        for x, y, z in zip(_left_attr_cols_name, _right_attr_cols_name, attr_cols)
    ]

    new_df = _df_old.join(_df_new, on=join_key, how="outer")

    return new_df.select(*join_key, *_coalesce)
