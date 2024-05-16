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

# pylint: skip-file
# flake8: noqa

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from feature_generation.v1.core.aggregation.pivot import pivot_on_multiple_fields

# skip the pytest if regex not installed
pytest.importorskip("regex")

SparkSession.builder.config("spark.sql.shuffle.partitions", 1).getOrCreate()


@pytest.mark.parametrize(
    "pivot_fields, aggregation, name_prefix, name_suffix, expected_cols_and_sums, test_case_name",
    [
        (
            {"channel_cd": ["1", "2"]},
            {"unq_interaction": f.countDistinct("interaction_id")},
            "fea_",
            "_val",
            {"fea_unq_interaction_1_val": 7, "fea_unq_interaction_2_val": 1},
            "Single field, single aggregation",
        ),
        (
            {"channel_cd": None},
            {"unq_interaction": f.countDistinct("interaction_id")},
            "fea_",
            "",
            {"fea_unq_interaction_1": 7, "fea_unq_interaction_2": 1},
            "Field values need to be discovered automatically",
        ),
        (
            {"channel_cd": ["1"], "product_cd": ["asp", "pan"]},
            {"unq_interaction": f.countDistinct("interaction_id")},
            "fea_",
            "_val",
            {"fea_unq_interaction_1_asp_val": 5, "fea_unq_interaction_1_pan_val": 2},
            "Multiple fields",
        ),
        (
            {"product_cd": ["asp"], "key_message_cd": ["^bla$", "bla2", "bla3"]},
            {
                "unq_interaction": f.countDistinct("interaction_id"),
                "cnt_all": f.count("*"),
            },
            "fea_",
            "_val",
            {
                "fea_unq_interaction_bla_asp_val": 3,
                "fea_unq_interaction_bla2_asp_val": 2,
                "fea_unq_interaction_bla3_asp_val": 2,
                "fea_cnt_all_bla_asp_val": 3,
                "fea_cnt_all_bla2_asp_val": 2,
                "fea_cnt_all_bla3_asp_val": 2,
            },
            "Multiple aggregations",
        ),
    ],
)
def test_pivot_one(
    mock_interactions_df,
    mock_key_messages_df,
    pivot_fields,
    aggregation,
    name_prefix,
    name_suffix,
    expected_cols_and_sums,
    test_case_name,
):
    group_fields = ["customer_id", "interaction_dt_last_day"]

    interactions_df = mock_interactions_df.withColumn(
        "interaction_dt_last_day", f.last_day(f.col("interaction_dt"))
    )

    interaction_km = interactions_df.join(
        mock_key_messages_df, how="left", on="interaction_id"
    )

    pivoted_df = pivot_on_multiple_fields(
        df=interaction_km,
        group_by=group_fields,
        pivot_fields=pivot_fields,
        aggregations=aggregation,
        name_prefix=name_prefix,
        name_suffix=name_suffix,
    )

    pivoted_df.show()

    expected_cols = list(expected_cols_and_sums.keys())

    assert set(pivoted_df.columns) == (
        set(group_fields) | set(expected_cols)
    ), test_case_name

    for expected_col in expected_cols:
        assert (
            pivoted_df.agg(f.sum(expected_col)).collect()[0][0]
            == expected_cols_and_sums[expected_col]
        ), test_case_name
