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
from pyspark.sql import functions as f

from feature_generation.v1.core.features.create_column import create_columns_from_config
from feature_generation.v1.core.features.flags import (
    arrays_overlap,
    coalesce,
    expr_col,
    expr_flag,
    nth_occurrence,
    null,
    regexp_extract,
    rlike,
    tag_arrays_overlap,
)


def test_create_flags_from_config_with_sub_flags(
    get_sample_spark_data_frame, get_sample_column_instructions
):  # pylint: disable=invalid-name
    df = get_sample_spark_data_frame
    df_with_flags = create_columns_from_config(
        df=df, column_instructions=get_sample_column_instructions
    )

    assert "data_related_hardcoded_flag" in df_with_flags.columns
    assert "data_related_occupation_based_flag" in df_with_flags.columns
    assert "engineer_related_occupation_based_flag" in df_with_flags.columns
    assert (
        "engineer_related_occupation_based_flag_matched_value" in df_with_flags.columns
    )
    assert "knight_of_the_seven_realms_flag" in df_with_flags.columns
    assert (
        df_with_flags.select(f.sum(f.col("data_related_hardcoded_flag"))).collect()[0][
            0
        ]
        == 1
    )
    assert (
        df_with_flags.select(
            f.sum(f.col("data_related_occupation_based_flag"))
        ).collect()[0][0]
        == 4
    )
    assert (
        df_with_flags.select(
            f.sum(f.col("engineer_related_occupation_based_flag"))
        ).collect()[0][0]
        == 3
    )
    assert (
        df_with_flags.select(f.sum(f.col("knight_of_the_seven_realms_flag"))).collect()[
            0
        ][0]
        == 0
    )
    assert "knight_of_the_seven_realms_flag_knight_match" in df_with_flags.columns
    assert [
        x[0]
        for x in df_with_flags.select(
            f.col("data_related_occupation_based_age_col")
        ).collect()
    ] == [31, 12, 65, 0, 0, 0, 26, 0]


@pytest.mark.parametrize(
    "values, input, output, expected_output",
    [
        ([1], "number", "is_one_flag", 1),
        ([3], "number", "is_three_flag", 3),
        ([8, 9], "number", "is_eight_or_nine_flag", 4),
        (["House Stark"], "house", "is_stark_flag", 4),
        (["House Lannister"], "house", "is_lannister_flag", 4),
        (["House Lannister", "House Stark"], "house", "is_lannister_or_stark_flag", 6),
    ],
)
def test_arrays_overlap_integer(
    values, input, output, expected_output, get_sample_spark_data_frame
):
    df = get_sample_spark_data_frame

    df_with_flags = df.withColumn(
        output, arrays_overlap(input=input, values=values, output=output)
    )

    assert df_with_flags.select(f.sum(f.col(output))).collect()[0][0] == expected_output


def test_tag_arrays_overlap(mock_event_df):
    result = mock_event_df.withColumn(
        "test", tag_arrays_overlap(input="tags", values="visit_gp", output="test")
    )

    assert [x[0] for x in result.select("test").collect()] == [0, 0, 1, 0, 0]


def test_null(mock_numbers_df):
    df = create_columns_from_config(mock_numbers_df, [null("num2", True, "num2_ind")])
    assert df.filter(f.col("id") == f.lit("id_4")).collect()[0]["num2_ind"] == 1
    assert df.filter(f.col("id") == f.lit("id_2")).collect()[0]["num2_ind"] == 0


def test_nnull(mock_numbers_df):
    df = create_columns_from_config(mock_numbers_df, [null("num2", False, "num2_ind")])
    assert df.filter(f.col("id") == f.lit("id_4")).collect()[0]["num2_ind"] == 0
    assert df.filter(f.col("id") == f.lit("id_2")).collect()[0]["num2_ind"] == 1


def test_nth_occurrence(mock_event_df):
    cfg = [
        nth_occurrence(
            "fever_first_occurrence_flag",
            "tags",
            "had_fever",
            {
                "partition_by": "person",
                "order_by": "date_index",
            },
            1,
        )
    ]

    df = create_columns_from_config(mock_event_df, cfg)

    assert (
        df.filter(f.col("fever_first_occurrence_flag") == f.lit(1))
        .select("date_index")
        .collect()[0][0]
        == 2
    )


def test_regexp_extract(get_sample_spark_data_frame):
    cfg = [regexp_extract("name", [".*e.*"], "name_contains_e")]

    df = create_columns_from_config(get_sample_spark_data_frame, cfg)

    results = [
        x[0]
        for x in df.filter(f.col("name_contains_e").isNotNull())
        .select("name_contains_e")
        .collect()
    ]
    results.sort()

    assert results == ["Cersei", "Daenerys", "Gendry", "Jaime"]


def test_expr_col(mock_numbers_df, mock_arrays_df):
    case_expr = "case when num1>num2 then num1-num2 else coalesce(num1, num2) end"
    df1 = mock_numbers_df.select("*", expr_col(expr=case_expr, output="new_column"))
    new_cols = [row[0] for row in df1.orderBy("id").select("new_column").collect()]
    assert new_cols == [1.0, 2.3, 12.1, 5, 0, 20.5]

    size_expr = "size(value)"
    df2 = mock_arrays_df.select("*", expr_col(expr=size_expr, output="size"))
    size = df2.select("size").collect()[0][0]
    assert size == 5


def test_expr_flag(mock_numbers_df, mock_arrays_df):
    expr1 = "num1>num2"
    df1 = mock_numbers_df.select("*", expr_flag(expr=expr1, output="flag"))
    flags = [row[0] for row in df1.orderBy("id").select("flag").collect()]
    assert flags == [True, False, True, False, False, False]

    expr2 = "size(value)>5"
    df2 = mock_arrays_df.select("*", expr_flag(expr=expr2, output="flag2"))
    flag2 = df2.select("flag2").collect()[0][0]
    assert not flag2


def test_coalesce_flag(mock_numbers_df):
    df1 = mock_numbers_df.select("*", coalesce(output="flags", cols=["num1", "num2"]))
    flags = [row[0] for row in df1.orderBy("id").select("flags").collect()]
    assert flags == [1, 2.3, 10, 5, 0, 20.5]


@pytest.mark.parametrize(
    "regex, flag_result_list, input_col_name",
    [
        (["^[A-Za-z]"], [1, 1, 0, 0, 0, 1], "sample_col_1"),
        (["^[0-9]"], [1, 1, 1, 0, 1, 0], "sample_col_2"),
        (["^2", "^3"], [0, 1, 1, 0, 1, 0], "sample_col_2"),
        ("^[A-Za-z]", [1, 1, 0, 0, 0, 1], "sample_col_1"),
    ],
)
def test_rlike_flag(mock_strings_df, regex, flag_result_list, input_col_name):
    df1 = mock_strings_df.select(
        "*",
        rlike(input=input_col_name, values=regex, output="rlike_flag1"),
    )
    rlike_flag1 = [row[0] for row in df1.select("rlike_flag1").collect()]

    assert rlike_flag1 == flag_result_list


@pytest.mark.parametrize(
    "regex, flag_result_list, input_col_name",
    [({"regexp": "^[A-Za-z]"}, TypeError, "sample_col_1")],
)
def test_rlike_flag_exception(mock_strings_df, regex, flag_result_list, input_col_name):
    with pytest.raises(TypeError):
        mock_strings_df.select(
            "*", rlike(input=input_col_name, values=regex, output="rlike_flag1")
        )
