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
"""Pivot function."""

import itertools
from typing import Any, Iterable, List, Mapping

import pyspark
import pyspark.sql.functions as f
import regex as re

_PIVOT_COMPOSITE_KEY = "_pivot_composite_key"


def pivot_on_multiple_fields(  # pylint: disable=too-many-locals
    df: pyspark.sql.DataFrame,
    group_by: List[str],
    pivot_fields: Mapping[str, Iterable[Any]],
    aggregations: Mapping[str, pyspark.sql.Column],
    name_prefix: str = "",
    name_suffix: str = "",
) -> pyspark.sql.DataFrame:
    """Creates aggregations on pivot tables.

    Groups a DataFrame by fields in `group_by`, pivots by keys (fields) `pivot_on`,
    and then creates aggregation columns in the `aggregations` dict.

    Names are always deterministic and follow the following
    pattern: {prefix}{agg_name}_{key1}_{key2}_{keyN}{suffix}.
    It's important to mention that column names are sorted to
    guarantee the deterministic behaviour.
    In the example above, key1 and key2 are sorted.

    Args:
        df: A DataFrame to pivot.
        group_by: A list of names of fields to use as indices.
        pivot_fields: A mapping from field names to a
            list of possible values to consider. If a list of possible values is
            null or empty for a field, use all values found in `df`.
        aggregations: A list of aggregation columns to run on the pivot table.
        name_prefix: prefix that will be added to a column name
        name_suffix: suffix that will be added to a column name

    Returns:
        A pivoted DataFrame with the new pivot aggregations.
    """
    distinct_values_pivot_fields = []
    pivot_field_names = sorted(pivot_fields.keys())

    for field in pivot_field_names:
        if len(pivot_fields[field] or []) == 0:
            distinct_values_pivot_fields += [
                [
                    row[field]
                    for row in df.select(field).distinct().collect()
                    if row[field] is not None
                ]
            ]
        else:
            assert len(set(pivot_fields[field])) == len(
                pivot_fields[field]
            ), f"Only unique values for field <{field}> are accepted"

            regex_exp = "|".join(pivot_fields[field])
            distinct_values_pivot_fields += [
                [
                    row[field]
                    for row in df.select(field).distinct().collect()
                    if re.match(regex_exp, row[field])
                ]
            ]

    # Generate cartesian product of pivot fields.
    cartesian_product_pivot_fields = itertools.product(*distinct_values_pivot_fields)

    # Create a new composite key on the cartesian product of pivot fields.
    df_with_composite_key = df.withColumn(
        _PIVOT_COMPOSITE_KEY,
        f.concat_ws("_", *[f.col(field) for field in pivot_field_names]),
    )

    aggregation_columns = []
    for ouput_col_name, aggregation in aggregations.items():
        aggregation_columns.append(aggregation.alias(ouput_col_name))

    # Spark ignores the aggregation alias when applying only one aggregation,
    # so we duplicate that aggregation to work around this.
    if len(aggregation_columns) == 1:
        aggregation_columns += [aggregation_columns[0].alias("::")]

    list_of_cartesian_pairs = list(map("_".join, cartesian_product_pivot_fields))

    # Pivot over the composite key and compute aggregations.
    df_pivot = (
        df_with_composite_key.groupBy(group_by)
        .pivot(_PIVOT_COMPOSITE_KEY, list_of_cartesian_pairs)
        .agg(*aggregation_columns)
    )

    # spark bug
    df_pivot = df_pivot.select([x for x in df_pivot.columns if not x.endswith("::")])

    for field_name in list(set(df_pivot.columns) - set(group_by)):
        df_pivot = df_pivot.withColumnRenamed(
            field_name,
            "{}{}{}".format(  # pylint: disable=consider-using-f-string # noqa: E501
                name_prefix,
                _construct_name(field_name, list_of_cartesian_pairs),
                name_suffix,
            ),
        )

    return df_pivot


def _construct_name(fieldname: str, list_of_cartesian_pairs: Iterable[str]) -> str:
    """Constructs final deterministic name.

    Puts aggregation alias in the beggining from the end.

    Args:
        fieldname: which column to construct for
        list_of_cartesian_pairs: list of all pairs to look in

    Returns:
        Final name

    Raises:
        ValueError: throws if something is wrong
    """
    for pair in list_of_cartesian_pairs:
        if fieldname.startswith(f"{pair}_"):
            agg_alias = fieldname.replace(f"{pair}_", "", 1)
            return f"{agg_alias}_{pair}"

    raise ValueError("Seems to be something wrong")
