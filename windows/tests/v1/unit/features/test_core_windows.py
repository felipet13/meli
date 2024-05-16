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
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.window import WindowSpec

from feature_generation.v1.core.features.create_column import create_columns_from_config
from feature_generation.v1.core.features.flags import expr_flag
from feature_generation.v1.core.features.windows import (
    aggregate_over_slice_grid,
    generate_array_elements_window_grid,
    generate_distinct_element_window_grid,
    generate_window_delta,
    generate_window_grid,
    generate_window_ratio,
    generate_windows_spec,
    window_column,
)
from feature_generation.v1.core.timeseries.array_aggregate import (
    array_max,
    array_mean,
    array_sum,
)


def test_window_feature_from_config(_get_sample_spark_data_frame2):
    df = _get_sample_spark_data_frame2

    column_instructions = [
        window_column(
            "x_flag_sum_last_3d",
            f.sum("x_flag"),
            Window.partitionBy("name").orderBy("date_index").rangeBetween(-3, -1),
        ),
        window_column(
            ["x_flag_mean_last_2d", "x_flag_mean_last_3d"],
            f.mean("x_flag"),
            [
                Window.partitionBy("name").orderBy("date_index").rowsBetween(-2, -2),
                Window.partitionBy("name").orderBy("date_index").rowsBetween(-3, -3),
            ],
        ),
        window_column(
            "x_flag_min_last_5d",
            f.min("x_flag"),
            Window.partitionBy("name").orderBy("date_index").rangeBetween(-5, -1),
        ),
        window_column(
            "x_flag_sum_last_3events",
            f.sum("x_flag"),
            [Window.partitionBy("name").orderBy("date_index").rowsBetween(-3, -1)],
        ),
        window_column(
            "x_flag_sum_past",
            f.sum("x_flag"),
            Window.partitionBy("name")
            .orderBy("date_index")
            .rowsBetween(Window.unboundedPreceding, -1),
        ),
    ]

    df_with_window_features = create_columns_from_config(df, column_instructions)

    cersei_latest_3days = (
        df_with_window_features.filter(
            (f.col("name") == "Cersei") & (f.col("date_index") == 15475)
        )
        .select("x_flag_sum_last_3d")
        .collect()[0][0]
    )

    cersei_latest_3events = (
        df_with_window_features.filter(
            (f.col("name") == "Cersei") & (f.col("date_index") == 15475)
        )
        .select("x_flag_sum_last_3events")
        .collect()[0][0]
    )

    gendry_past_events = (
        df_with_window_features.filter(
            (f.col("name") == "Gendry") & (f.col("date_index") == 15465)
        )
        .select("x_flag_sum_past")
        .collect()[0][0]
    )

    assert "x_flag_sum_last_3d" in df_with_window_features.columns
    assert "x_flag_mean_last_2d" in df_with_window_features.columns
    assert "x_flag_mean_last_3d" in df_with_window_features.columns
    assert "x_flag_min_last_5d" in df_with_window_features.columns
    assert "x_flag_sum_last_3events" in df_with_window_features.columns
    assert cersei_latest_3days == 2
    assert cersei_latest_3events == 3
    assert gendry_past_events == 2


def test_window_spec_no_orderby(spark):
    window = generate_windows_spec(partition_by=["name"])
    assert isinstance(window[0], WindowSpec)


def test_window_spec_no_ranges(spark):
    window = generate_windows_spec(
        partition_by=["name"],
        order_by="date_index",
    )
    assert isinstance(window[0], WindowSpec)


def test_valid_output_and_ranges(_get_sample_spark_data_frame2):
    df = _get_sample_spark_data_frame2

    column_instructions = [
        window_column(
            "x_flag_min_last_2_2d",
            f.min("x_flag"),
            [Window.partitionBy("name").orderBy("date_index").rangeBetween(-2, -2)],
        ),
        window_column(
            "y_flag_sum", f.sum("y_flag"), generate_windows_spec(partition_by="name")
        ),
    ]

    result = create_columns_from_config(
        df,
        column_instructions,
    )

    assert "y_flag_sum" in result.columns
    assert "x_flag_min_last_2_2d" in result.columns


@pytest.fixture
def grid_dict():
    grid_dict = {
        "inputs": ["x_flag", "y_flag"],
        "funcs": [f.max, f.sum],
        "windows": [
            {"partition_by": ["name"], "order_by": ["date_index"], "descending": False},
            {"partition_by": ["name"], "order_by": ["date_index"], "descending": True},
        ],
        "ranges_between": [
            [-1, -1],
            [0, 0],
            [1, 1],
            [-2, 0],
            [0, 4],
            [-1, "UNBOUNDED FOLLOWING"],
            ["unbounded preceding", 0],
        ],
        "negative_term": "past",
        "positive_term": "next",
        "prefix": "ftr_",
        "suffix": "d",
    }
    return grid_dict


@pytest.fixture
def assert_generated_columns():
    return [
        "name",
        "date",
        "date_index",
        "x_flag",
        "y_flag",
        "ftr_x_flag_max_past_1_past_1d",
        "ftr_x_flag_max_between_0_and_0d",
        "ftr_x_flag_max_next_1_next_1d",
        "ftr_x_flag_max_past_2_next_0d",
        "ftr_x_flag_max_past_0_next_4d",
        "ftr_x_flag_max_past_1_next_unbounded_following_d",
        "ftr_x_flag_max_past_unbounded_preceding_next_0d",
        "ftr_x_flag_sum_past_1_past_1d",
        "ftr_x_flag_sum_between_0_and_0d",
        "ftr_x_flag_sum_next_1_next_1d",
        "ftr_x_flag_sum_past_2_next_0d",
        "ftr_x_flag_sum_past_0_next_4d",
        "ftr_x_flag_sum_past_1_next_unbounded_following_d",
        "ftr_x_flag_sum_past_unbounded_preceding_next_0d",
        "ftr_x_flag_max_past_1_past_1d",
        "ftr_x_flag_max_between_0_and_0d",
        "ftr_x_flag_max_next_1_next_1d",
        "ftr_x_flag_max_past_2_next_0d",
        "ftr_x_flag_max_past_0_next_4d",
        "ftr_x_flag_max_past_1_next_unbounded_following_d",
        "ftr_x_flag_max_past_unbounded_preceding_next_0d",
        "ftr_x_flag_sum_past_1_past_1d",
        "ftr_x_flag_sum_between_0_and_0d",
        "ftr_x_flag_sum_next_1_next_1d",
        "ftr_x_flag_sum_past_2_next_0d",
        "ftr_x_flag_sum_past_0_next_4d",
        "ftr_x_flag_sum_past_1_next_unbounded_following_d",
        "ftr_x_flag_sum_past_unbounded_preceding_next_0d",
        "ftr_y_flag_max_past_1_past_1d",
        "ftr_y_flag_max_between_0_and_0d",
        "ftr_y_flag_max_next_1_next_1d",
        "ftr_y_flag_max_past_2_next_0d",
        "ftr_y_flag_max_past_0_next_4d",
        "ftr_y_flag_max_past_1_next_unbounded_following_d",
        "ftr_y_flag_max_past_unbounded_preceding_next_0d",
        "ftr_y_flag_sum_past_1_past_1d",
        "ftr_y_flag_sum_between_0_and_0d",
        "ftr_y_flag_sum_next_1_next_1d",
        "ftr_y_flag_sum_past_2_next_0d",
        "ftr_y_flag_sum_past_0_next_4d",
        "ftr_y_flag_sum_past_1_next_unbounded_following_d",
        "ftr_y_flag_sum_past_unbounded_preceding_next_0d",
        "ftr_y_flag_max_past_1_past_1d",
        "ftr_y_flag_max_between_0_and_0d",
        "ftr_y_flag_max_next_1_next_1d",
        "ftr_y_flag_max_past_2_next_0d",
        "ftr_y_flag_max_past_0_next_4d",
        "ftr_y_flag_max_past_1_next_unbounded_following_d",
        "ftr_y_flag_max_past_unbounded_preceding_next_0d",
        "ftr_y_flag_sum_past_1_past_1d",
        "ftr_y_flag_sum_between_0_and_0d",
        "ftr_y_flag_sum_next_1_next_1d",
        "ftr_y_flag_sum_past_2_next_0d",
        "ftr_y_flag_sum_past_0_next_4d",
        "ftr_y_flag_sum_past_1_next_unbounded_following_d",
        "ftr_y_flag_sum_past_unbounded_preceding_next_0d",
        "test_multi_compabitibility_with_grid_flag",
    ]


def test_generate_window_grid_with_columns_from_config(
    _get_sample_spark_data_frame2, grid_dict, assert_generated_columns
):
    df = _get_sample_spark_data_frame2

    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            generate_window_grid(**grid_dict),
            expr_flag(
                expr="x_flag == 0", output="test_multi_compabitibility_with_grid_flag"
            ),
        ],
    )

    len_expanded_grid = (
        len(grid_dict["inputs"])
        * len(grid_dict["windows"])
        * len(grid_dict["funcs"])
        * len(grid_dict["ranges_between"])
    )

    assert 56 == len_expanded_grid

    assert (
        len(df_with_window_features.columns) == len_expanded_grid + len(df.columns) + 1
    )

    assert df_with_window_features.columns == assert_generated_columns


@pytest.fixture
def grid_dict_rank():
    grid_dict = {
        "inputs": ["x_flag", "y_flag"],
        "funcs": [f.rank, f.row_number, f.dense_rank, f.percent_rank],
        "windows": [
            {"partition_by": ["name"], "order_by": ["date_index"], "descending": False},
        ],
        "rows_between": [
            ["unbounded preceding", 0],
        ],
        "negative_term": "past",
        "positive_term": "next",
        "prefix": "ftr_",
        "suffix": "d",
    }
    return grid_dict


@pytest.fixture
def assert_generated_columns_rank():
    return [
        "name",
        "date",
        "date_index",
        "x_flag",
        "y_flag",
        "ftr_x_flag_rank_past_unbounded_preceding_next_0d",
        "ftr_x_flag_row_number_past_unbounded_preceding_next_0d",
        "ftr_x_flag_dense_rank_past_unbounded_preceding_next_0d",
        "ftr_x_flag_percent_rank_past_unbounded_preceding_next_0d",
        "ftr_y_flag_rank_past_unbounded_preceding_next_0d",
        "ftr_y_flag_row_number_past_unbounded_preceding_next_0d",
        "ftr_y_flag_dense_rank_past_unbounded_preceding_next_0d",
        "ftr_y_flag_percent_rank_past_unbounded_preceding_next_0d",
        "test_multi_compabitibility_with_grid_flag",
    ]


def test_generate_window_grid_with_columns_from_config_rank(
    _get_sample_spark_data_frame2, grid_dict_rank, assert_generated_columns_rank
):
    df = _get_sample_spark_data_frame2

    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            generate_window_grid(**grid_dict_rank),
            expr_flag(
                expr="x_flag == 0", output="test_multi_compabitibility_with_grid_flag"
            ),
        ],
    )

    len_expanded_grid = (
        len(grid_dict_rank["inputs"])
        * len(grid_dict_rank["windows"])
        * len(grid_dict_rank["funcs"])
        * len(grid_dict_rank["rows_between"])
    )

    arya_rank = (
        df_with_window_features.filter(
            (f.col("date_index") == 15467) & (f.col("name") == "Arya")
        )
        .select(f.col("ftr_x_flag_rank_past_unbounded_preceding_next_0d"))
        .collect()[0][0]
    )
    arya_row_number = (
        df_with_window_features.filter(
            (f.col("date_index") == 15467) & (f.col("name") == "Arya")
        )
        .select(f.col("ftr_x_flag_row_number_past_unbounded_preceding_next_0d"))
        .collect()[0][0]
    )
    arya_dense_rank = (
        df_with_window_features.filter(
            (f.col("date_index") == 15467) & (f.col("name") == "Arya")
        )
        .select(f.col("ftr_x_flag_dense_rank_past_unbounded_preceding_next_0d"))
        .collect()[0][0]
    )
    arya_percent_rank = (
        df_with_window_features.filter(
            (f.col("date_index") == 15467) & (f.col("name") == "Arya")
        )
        .select(f.col("ftr_x_flag_percent_rank_past_unbounded_preceding_next_0d"))
        .collect()[0][0]
    )

    assert arya_rank == 2
    assert arya_row_number == 2
    assert arya_dense_rank == 2
    assert arya_percent_rank == 0.25

    assert 8 == len_expanded_grid

    assert len(df_with_window_features.columns) == 14

    assert df_with_window_features.columns == assert_generated_columns_rank


@pytest.fixture
def assert_generated_delta_columns():
    return [
        "name",
        "date",
        "date_index",
        "x_flag",
        "y_flag",
        "ftr_x_flag_sum_past_1_next_0d",
        "ftr_x_flag_sum_past_1_next_unbounded_following_d",
        "ftr_x_flag_sum_next_1_next_2d",
        "ftr_delta_x_flag_sum_past_1_next_unbounded_following_d_and_past_1_next_0d",
        "ftr_delta_x_flag_sum_next_1_next_2d_and_past_1_next_unbounded_following_d",
        "ftr_y_flag_sum_past_1_next_0d",
        "ftr_y_flag_sum_past_1_next_unbounded_following_d",
        "ftr_y_flag_sum_next_1_next_2d",
        "ftr_delta_y_flag_sum_past_1_next_unbounded_following_d_and_past_1_next_0d",
        "ftr_delta_y_flag_sum_next_1_next_2d_and_past_1_next_unbounded_following_d",
        "test_multi_compabitibility_with_grid_flag",
    ]


@pytest.fixture
def grid_delta_dict():
    grid_delta_dict = {
        "inputs": ["x_flag", "y_flag"],
        "funcs": [f.sum],
        "windows": [
            {"partition_by": ["name"], "order_by": ["date_index"], "descending": False},
        ],
        "ranges_between": [[-1, 0], [-1, "UNBOUNDED FOLLOWING"], [1, 2]],
        "negative_term": "past",
        "positive_term": "next",
        "prefix": "ftr_",
        "suffix": "d",
    }
    return grid_delta_dict


def test_generate_window_delta_with_columns_from_config(
    _get_sample_spark_data_frame2, grid_delta_dict, assert_generated_delta_columns
):
    df = _get_sample_spark_data_frame2

    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            generate_window_delta(**grid_delta_dict),
            expr_flag(
                expr="x_flag == 0", output="test_multi_compabitibility_with_grid_flag"
            ),
        ],
    )

    arya_ratio_x_1_1d = (
        df_with_window_features.filter(
            (f.col("date_index") == 15466) & (f.col("name") == "Arya")
        )
        .select(
            "ftr_delta_x_flag_sum_past_1_next_unbounded_following_d_and_past_1_next_0d"
        )
        .collect()[0][0]
    )
    arya_ratio_x_0_0d = (
        df_with_window_features.filter(
            (f.col("date_index") == 15466) & (f.col("name") == "Arya")
        )
        .select(
            "ftr_delta_x_flag_sum_next_1_next_2d_and_past_1_next_unbounded_following_d"
        )
        .collect()[0][0]
    )
    arya_ratio_15470 = (
        df_with_window_features.filter(
            (f.col("date_index") == 15470) & (f.col("name") == "Arya")
        )
        .select(
            "ftr_delta_x_flag_sum_next_1_next_2d_and_past_1_next_unbounded_following_d"
        )
        .collect()[0][0]
    )

    assert arya_ratio_x_1_1d == 2
    assert arya_ratio_x_0_0d == -3
    assert arya_ratio_15470 is None

    len_expanded_grid = (
        len(grid_delta_dict["inputs"])
        * len(grid_delta_dict["windows"])
        * len(grid_delta_dict["funcs"])
        * len(grid_delta_dict["ranges_between"])
    )

    assert 6 == len_expanded_grid

    assert (
        len(df_with_window_features.columns)
        == len_expanded_grid + len(df.columns) + 1 + 4
    )

    assert df_with_window_features.columns == assert_generated_delta_columns


@pytest.fixture
def assert_generated_grid_column_abbreviation():
    return [
        "name",
        "date",
        "date_index",
        "x_flag",
        "y_flag",
        "ftr_x_flag_sum_p_1_n_0d",
        "ftr_x_flag_sum_b_0_a_0d",
        "ftr_y_flag_sum_p_1_n_0d",
        "ftr_y_flag_sum_b_0_a_0d",
        "test_multi_compabitibility_with_grid_flag",
    ]


@pytest.fixture
def grid_abbreviation_dict():
    grid_abbreviation_dict = {
        "inputs": ["x_flag", "y_flag"],
        "funcs": [f.sum],
        "windows": [
            {"partition_by": ["name"], "order_by": ["date_index"], "descending": False},
        ],
        "ranges_between": [[-1, 0], [0, 0]],
        "negative_term": "p",
        "positive_term": "n",
        "negative_default": "b",
        "positive_default": "a",
        "prefix": "ftr_",
        "suffix": "d",
    }
    return grid_abbreviation_dict


def test_generated_grid_column_abbreviation(
    _get_sample_spark_data_frame2,
    grid_abbreviation_dict,
    assert_generated_grid_column_abbreviation,
):
    df = _get_sample_spark_data_frame2

    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            generate_window_grid(**grid_abbreviation_dict),
            expr_flag(
                expr="x_flag == 0", output="test_multi_compabitibility_with_grid_flag"
            ),
        ],
    )

    assert df_with_window_features.columns == assert_generated_grid_column_abbreviation


@pytest.fixture
def assert_generated_window_delta_column_abbreviation():
    return [
        "name",
        "date",
        "date_index",
        "x_flag",
        "y_flag",
        "ftr_x_flag_sum_p_2_n_0d",
        "ftr_x_flag_sum_b_0_a_0d",
        "ftr_delta_x_flag_sum_b_0_a_0d_and_p_2_n_0d",
        "ftr_y_flag_sum_p_2_n_0d",
        "ftr_y_flag_sum_b_0_a_0d",
        "ftr_delta_y_flag_sum_b_0_a_0d_and_p_2_n_0d",
        "test_multi_compabitibility_with_grid_flag",
    ]


@pytest.fixture
def delta_abbreviation_dict():
    delta_abbreviation_dict = {
        "inputs": ["x_flag", "y_flag"],
        "funcs": [f.sum],
        "windows": [
            {"partition_by": ["name"], "order_by": ["date_index"], "descending": False},
        ],
        "ranges_between": [[-2, 0], [0, 0]],
        "negative_term": "p",
        "positive_term": "n",
        "negative_default": "b",
        "positive_default": "a",
        "prefix": "ftr_",
        "suffix": "d",
    }
    return delta_abbreviation_dict


def test_generated_delta_column_abbreviation(
    _get_sample_spark_data_frame2,
    delta_abbreviation_dict,
    assert_generated_window_delta_column_abbreviation,
):
    df = _get_sample_spark_data_frame2

    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            generate_window_delta(**delta_abbreviation_dict),
            expr_flag(
                expr="x_flag == 0", output="test_multi_compabitibility_with_grid_flag"
            ),
        ],
    )

    assert (
        df_with_window_features.columns
        == assert_generated_window_delta_column_abbreviation
    )


@pytest.fixture
def assert_generated_window_ratio_column_abbreviation():
    return [
        "name",
        "date",
        "date_index",
        "x_flag",
        "y_flag",
        "ftr_x_flag_sum_p_3_n_0d",
        "ftr_x_flag_sum_b_0_a_0d",
        "ftr_y_flag_sum_p_3_n_0d",
        "ftr_y_flag_sum_b_0_a_0d",
        "test_multi_compabitibility_with_grid_flag",
    ]


@pytest.fixture
def ratio_abbreviation_dict():
    ratio_abbreviation_dict = {
        "inputs": ["x_flag", "y_flag"],
        "funcs": [f.sum],
        "windows": [
            {"partition_by": ["name"], "order_by": ["date_index"], "descending": False},
        ],
        "ranges_between": [[-3, 0], [0, 0]],
        "negative_term": "p",
        "positive_term": "n",
        "negative_default": "b",
        "positive_default": "a",
        "prefix": "ftr_",
        "suffix": "d",
    }
    return ratio_abbreviation_dict


def test_generated_ratio_column_abbreviation(
    _get_sample_spark_data_frame2,
    ratio_abbreviation_dict,
    assert_generated_window_ratio_column_abbreviation,
):
    df = _get_sample_spark_data_frame2

    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            generate_window_grid(**ratio_abbreviation_dict),
            expr_flag(
                expr="x_flag == 0", output="test_multi_compabitibility_with_grid_flag"
            ),
        ],
    )

    assert (
        df_with_window_features.columns
        == assert_generated_window_ratio_column_abbreviation
    )


@pytest.fixture
def grid_delta_dict_value_exception():
    grid_delta_dict_value_exception = {
        "inputs": ["x_flag"],
        "funcs": [f.sum, f.max],
        "windows": [
            {
                "partition_by": ["name"],
                "order_by": ["date_index"],
                "descending": False,
            },
        ],
        "ranges_between": [
            [-2, 0],
        ],
        "negative_term": "past",
        "positive_term": "next",
        "prefix": "ftr_",
        "suffix": "d",
    }
    return grid_delta_dict_value_exception


def test_generate_window_delta_with_exceptions(grid_delta_dict_value_exception):
    with pytest.raises(
        ValueError,
        match="""At least 2 window ranges should be provided.""",
    ):
        generate_window_delta(**grid_delta_dict_value_exception)


@pytest.fixture
def assert_generated_ratio_columns():
    return [
        "name",
        "date",
        "date_index",
        "x_flag",
        "y_flag",
        "total_flag",
        "ftr_ratio_x_flag_max_past_1_past_1d_total_flag_max_past_1_past_1d",
        "ftr_ratio_x_flag_max_between_0_and_0d_total_flag_max_between_0_and_0d",
        "ftr_ratio_x_flag_max_next_1_next_1d_total_flag_max_next_1_next_1d",
        "ftr_ratio_x_flag_sum_past_1_past_1d_total_flag_sum_past_1_past_1d",
        "ftr_ratio_x_flag_sum_between_0_and_0d_total_flag_sum_between_0_and_0d",
        "ftr_ratio_x_flag_sum_next_1_next_1d_total_flag_sum_next_1_next_1d",
        "ftr_ratio_y_flag_max_past_1_past_1d_total_flag_max_past_1_past_1d",
        "ftr_ratio_y_flag_max_between_0_and_0d_total_flag_max_between_0_and_0d",
        "ftr_ratio_y_flag_max_next_1_next_1d_total_flag_max_next_1_next_1d",
        "ftr_ratio_y_flag_sum_past_1_past_1d_total_flag_sum_past_1_past_1d",
        "ftr_ratio_y_flag_sum_between_0_and_0d_total_flag_sum_between_0_and_0d",
        "ftr_ratio_y_flag_sum_next_1_next_1d_total_flag_sum_next_1_next_1d",
        "ftr_x_flag_max_past_1_past_1d",
        "ftr_total_flag_max_past_1_past_1d",
        "ftr_x_flag_max_between_0_and_0d",
        "ftr_total_flag_max_between_0_and_0d",
        "ftr_x_flag_max_next_1_next_1d",
        "ftr_total_flag_max_next_1_next_1d",
        "ftr_x_flag_sum_past_1_past_1d",
        "ftr_total_flag_sum_past_1_past_1d",
        "ftr_x_flag_sum_between_0_and_0d",
        "ftr_total_flag_sum_between_0_and_0d",
        "ftr_x_flag_sum_next_1_next_1d",
        "ftr_total_flag_sum_next_1_next_1d",
        "ftr_y_flag_max_past_1_past_1d",
        "ftr_y_flag_max_between_0_and_0d",
        "ftr_y_flag_max_next_1_next_1d",
        "ftr_y_flag_sum_past_1_past_1d",
        "ftr_y_flag_sum_between_0_and_0d",
        "ftr_y_flag_sum_next_1_next_1d",
        "test_multi_compabitibility_with_grid_flag",
    ]


@pytest.fixture
def grid_ratio_dict():
    grid_ratio_dict = {
        "inputs": {"x_flag": "total_flag", "y_flag": "total_flag"},
        "funcs": [f.max, f.sum],
        "windows": [
            {"partition_by": ["name"], "order_by": ["date_index"], "descending": False},
        ],
        "ranges_between": [
            [-1, -1],
            [0, 0],
            [1, 1],
        ],
        "negative_term": "past",
        "positive_term": "next",
        "prefix": "ftr_",
        "suffix": "d",
    }
    return grid_ratio_dict


def test_generate_window_ratio_with_columns_from_config(
    _get_sample_spark_data_frame2, grid_ratio_dict, assert_generated_ratio_columns
):
    df = _get_sample_spark_data_frame2.withColumn(
        "total_flag", f.col("x_flag") + f.col("y_flag")
    )

    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            generate_window_ratio(**grid_ratio_dict),
            expr_flag(
                expr="x_flag == 0", output="test_multi_compabitibility_with_grid_flag"
            ),
        ],
    )

    arya_ratio_x_1_1d = (
        df_with_window_features.filter(
            (f.col("date_index") == 15466) & (f.col("name") == "Arya")
        )
        .select("ftr_ratio_x_flag_max_past_1_past_1d_total_flag_max_past_1_past_1d")
        .collect()[0][0]
    )
    arya_ratio_x_0_0d = (
        df_with_window_features.filter(
            (f.col("date_index") == 15466) & (f.col("name") == "Arya")
        )
        .select("ftr_ratio_x_flag_max_between_0_and_0d_total_flag_max_between_0_and_0d")
        .collect()[0][0]
    )
    arya_ratio_15470 = (
        df_with_window_features.filter(
            (f.col("date_index") == 15470) & (f.col("name") == "Arya")
        )
        .select("ftr_ratio_x_flag_max_between_0_and_0d_total_flag_max_between_0_and_0d")
        .collect()[0][0]
    )

    assert arya_ratio_x_1_1d is None
    assert arya_ratio_x_0_0d == 1.0
    assert arya_ratio_15470 == 0.3333333333333333

    len_expanded_grid = (
        len({col for inp in grid_ratio_dict["inputs"].items() for col in inp})
        * len(grid_ratio_dict["windows"])
        * len(grid_ratio_dict["funcs"])
        * len(grid_ratio_dict["ranges_between"])
    )

    assert 18 == len_expanded_grid

    assert (
        len(df_with_window_features.columns)
        == len_expanded_grid + len(df.columns) + 1 + 12
    )

    assert df_with_window_features.columns == assert_generated_ratio_columns


@pytest.fixture
def grid_ratio_dict_value_exception():
    grid_ratio_dict_value_exception = {
        "inputs": ["x_flag"],
        "funcs": [f.sum, f.max],
        "windows": [
            {
                "partition_by": ["name"],
                "order_by": ["date_index"],
                "descending": False,
            },
        ],
        "ranges_between": [
            [-1, 0],
        ],
        "negative_term": "past",
        "positive_term": "next",
        "prefix": "ftr_",
        "suffix": "d",
    }
    return grid_ratio_dict_value_exception


def test_generate_window_ratio_with_exceptions(
    grid_ratio_dict_value_exception,
):
    with pytest.raises(
        TypeError,
        match="""Argument ``inputs`` should be of dict type.""",
    ):
        generate_window_ratio(**grid_ratio_dict_value_exception)


@pytest.fixture
def assert_generated_distinct_element_columns():
    return [
        "npi_id",
        "observation_dt",
        "date_index",
        "all_patient",
        "male_patient",
        "female_patient",
        "ftr_all_patient_count_past_180_past_1d",
        "ftr_all_patient_count_past_90_past_1d",
        "ftr_all_patient_count_past_30_past_1d",
        "ftr_male_patient_count_past_180_past_1d",
        "ftr_male_patient_count_past_90_past_1d",
        "ftr_male_patient_count_past_30_past_1d",
        "ftr_female_patient_count_past_180_past_1d",
        "ftr_female_patient_count_past_90_past_1d",
        "ftr_female_patient_count_past_30_past_1d",
        "test_multi_compabitibility_with_grid_flag",
    ]


@pytest.fixture
def distinct_element_dict():
    distinct_element_dict = {
        "inputs": ["all_patient", "male_patient", "female_patient"],
        "windows": [
            {
                "partition_by": ["npi_id"],
                "order_by": ["date_index"],
                "descending": False,
            },
        ],
        "ranges_between": [[-180, -1], [-90, -1], [-30, -1]],
        "prefix": "ftr_",
        "suffix": "d",
    }
    return distinct_element_dict


def test_generate_distinct_element_window_grid_with_columns_from_config(
    _get_sample_spark_data_frame3,
    distinct_element_dict,
    assert_generated_distinct_element_columns,
):
    df = _get_sample_spark_data_frame3

    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            generate_distinct_element_window_grid(**distinct_element_dict),
            expr_flag(
                expr="size(all_patient) == 0",
                output="test_multi_compabitibility_with_grid_flag",
            ),
        ],
    )

    npi_1738_180_1d = (
        df_with_window_features.filter(
            (f.col("date_index") == 17896) & (f.col("npi_id") == "0000001738")
        )
        .select("ftr_female_patient_count_past_180_past_1d")
        .collect()[0][0]
    )

    npi_1738_90_1d = (
        df_with_window_features.filter(
            (f.col("date_index") == 17896) & (f.col("npi_id") == "0000001738")
        )
        .select("ftr_female_patient_count_past_90_past_1d")
        .collect()[0][0]
    )

    assert npi_1738_180_1d == 1
    assert npi_1738_90_1d == 0

    len_expanded_grid = (
        len(distinct_element_dict["inputs"])
        * len(distinct_element_dict["windows"])
        * len(distinct_element_dict["ranges_between"])
    )

    assert 9 == len_expanded_grid

    assert (
        len(df_with_window_features.columns) == len_expanded_grid + len(df.columns) + 1
    )

    assert df_with_window_features.columns == assert_generated_distinct_element_columns


def _get_sample_spark_data_frame(df_input_slice):
    npi_unbounded_window = (
        Window.partitionBy("npi_id")
        .orderBy("time_index")
        .rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )

    df_collect_list = df_input_slice.withColumn(
        "time_list", f.collect_list("time_index").over(npi_unbounded_window)
    ).withColumn("value_list", f.collect_list("value").over(npi_unbounded_window))

    return df_collect_list


@pytest.fixture
def grid_dict():
    grid_dict = {
        "inputs": ["value_list"],
        "funcs": [array_max, array_sum],
        "anchor_col": "time_index",
        "anchor_array": "time_list",
        "ranges_between": [
            [-1, -1],
            [0, 0],
            [1, 1],
        ],
        "prefix": "ftr_",
    }

    return grid_dict


@pytest.fixture
def assert_generated_columns():
    return [
        "npi_id",
        "time_index",
        "value",
        "time_list",
        "value_list",
        "ftr_value_list_array_max_past_1_past_1",
        "ftr_value_list_array_max_between_0_and_0",
        "ftr_value_list_array_max_next_1_next_1",
        "ftr_value_list_array_sum_past_1_past_1",
        "ftr_value_list_array_sum_between_0_and_0",
        "ftr_value_list_array_sum_next_1_next_1",
    ]


def test_generate_window_grid_with_columns_from_config(
    df_input_slice, grid_dict, assert_generated_columns
):
    df = _get_sample_spark_data_frame(df_input_slice)

    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            aggregate_over_slice_grid(
                **grid_dict,
            )
        ],
    )

    len_expanded_grid = (
        len(grid_dict["inputs"])
        * len(grid_dict["funcs"])
        * len(grid_dict["ranges_between"])
    )

    assert 6 == len_expanded_grid
    # 6 is used as a unique key for validating the length of
    # `inputs`(2), `funcs`(1), `ranges_between`(3) all in one assert.
    assert len(df_with_window_features.columns) == len_expanded_grid + len(df.columns)

    assert df_with_window_features.columns == assert_generated_columns


@pytest.fixture
def array_window_dict_1():
    array_window_dict = {
        "inputs": ["feature"],
        "windows": [
            {
                "partition_by": ["npi_id"],
                "order_by": ["week_index"],
                "descending": False,
            },
        ],
        "agg_functions": [array_mean, array_max],
        "ranges_between": [[-2, -1], [0, 2]],
        "prefix": "ftr_",
        "suffix": "w",
    }
    return array_window_dict


@pytest.fixture
def array_window_dict_2():
    array_window_dict = {
        "inputs": ["feature"],
        "windows": [
            {
                "partition_by": ["npi_id"],
                "order_by": ["week_index"],
                "descending": False,
            },
        ],
        "agg_functions": [array_mean, array_max],
        "prefix": "ftr_",
        "suffix": "w",
    }
    return array_window_dict


@pytest.fixture
def array_window_dict_3():
    array_window_dict = {
        "inputs": ["feature"],
        "windows": [
            {
                "partition_by": ["npi_id"],
                "order_by": ["week_index"],
                "descending": False,
            },
        ],
        "agg_functions": [array_mean, array_max],
        "ranges_between": [[-2, -1], [0, 2]],
        "rows_between": [[-2, -1], [0, 2]],
        "prefix": "ftr_",
        "suffix": "w",
    }
    return array_window_dict


def test_generate_array_elements_window_grid(
    _get_sample_spark_data_frame4,
    array_window_dict_1,
):
    df = _get_sample_spark_data_frame4
    df_with_window_features = create_columns_from_config(
        df=df,
        column_instructions=[
            generate_array_elements_window_grid(**array_window_dict_1),
        ],
    )

    # Asserting if the 2 features each aggregate function were created
    assert len(df_with_window_features.columns) == len(df.columns) + 4
    # Asserting if the calculated value is coherent with the expected value
    assert (
        df_with_window_features.select("ftr_feature_array_mean_past_0_next_2w")
        .filter("npi_id = '1' and week_index = 3")
        .collect()[0][0]
        == 48.0
    )
    assert (
        df_with_window_features.select("ftr_feature_array_mean_past_2_past_1w")
        .filter("npi_id = '1' and week_index = 3")
        .collect()[0][0]
        == 18.6
    )


def test_generate_array_elements_window_grid_with_no_range(
    _get_sample_spark_data_frame4,
    array_window_dict_2,
):
    df = _get_sample_spark_data_frame4
    with pytest.raises(
        ValueError, match="Please supply either ``ranges_between`` or ``rows_between``."
    ):
        df_with_window_features = create_columns_from_config(
            df=df,
            column_instructions=[
                generate_array_elements_window_grid(**array_window_dict_2),
            ],
        )


def test_generate_array_elements_window_grid_with_both_range(
    _get_sample_spark_data_frame4,
    array_window_dict_3,
):
    df = _get_sample_spark_data_frame4
    with pytest.raises(
        ValueError, match="Please supply either ``ranges_between`` or ``rows_between``."
    ):
        df_with_window_features = create_columns_from_config(
            df=df,
            column_instructions=[
                generate_array_elements_window_grid(**array_window_dict_3),
            ],
        )
