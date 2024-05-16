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

"""Test functions for the array related functions."""

# pylint: skip-file
# flake8: noqa
import pytest
from pyspark.sql import Window
from pyspark.sql import functions as f

from feature_generation.v1.core.timeseries.array_aggregate import (
    aggregate_over_slice,
    array_auc_time_delta,
    array_auc_trapezoidal,
    array_max,
    array_mean,
    array_min,
    array_stddev,
    array_sum,
    array_variance,
    array_weighted_mean,
    return_slice,
)


def test_array_mean(mock_arrays_df):
    mean = mock_arrays_df.select(
        array_mean(value="time_index", alias="mean")
    ).collect()[0][0]
    assert mean == 3.2


def test_array_sum(mock_arrays_df):
    sum = mock_arrays_df.select(array_sum(value="time_index", alias="sum")).collect()[
        0
    ][0]
    assert sum == 16


def test_array_variance(mock_arrays_df):
    mock_arrays_df = mock_arrays_df.select(
        "*", array_mean(value="time_index", alias="mean_c1").cast("int")
    )
    variance = mock_arrays_df.select(
        array_variance(value="time_index", mean="mean_c1", alias="variance")
    ).collect()[0][0]
    assert round(variance, 2) == 4.20


def test_array_stddev(mock_arrays_df):
    mock_arrays_df = mock_arrays_df.select(
        "*", array_mean(value="time_index", alias="mean_c1")
    )
    stddev = mock_arrays_df.select(
        array_stddev(value="time_index", mean="mean_c1", alias="stddev")
    ).collect()[0][0]
    assert round(stddev, 3) == 1.72


def test_weighted_mean(mock_arrays_df):
    weighted_mean = mock_arrays_df.select(
        array_weighted_mean(value="time_index", weight="value", alias="mean")
    ).collect()[0][0]
    assert weighted_mean == 3.25


def test_array_min(mock_arrays_df):
    minimum = mock_arrays_df.select(
        array_min(value="time_index", alias="min")
    ).collect()[0][0]
    assert minimum == 1


def test_array_max(mock_arrays_df):
    maximum = mock_arrays_df.select(
        array_max(value="time_index", alias="max")
    ).collect()[0][0]
    assert maximum == 6


def test_array_auc_trapezoidal(mock_arrays_df):
    auc = mock_arrays_df.select(
        array_auc_trapezoidal(value="value", time="time_index", alias="auc").alias(
            "auc"
        )
    ).collect()[0][0]
    assert auc == 9.0


def test_array_auc_time_delta(mock_arrays_df):
    auc_tdelta = mock_arrays_df.select(
        array_auc_time_delta(value="value", time_delta=2, alias="auc")
    ).collect()[0][0]
    assert auc_tdelta == 14.0


def _get_sample_spark_data_frame(df_input_slice):
    w = (
        Window.partitionBy("npi_id")
        .orderBy("time_index")
        .rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )

    df_collect_list = df_input_slice.withColumn(
        "time_list", f.collect_list("time_index").over(w)
    ).withColumn("value_list", f.collect_list("value").over(w))

    return df_collect_list


def test_aggregate_over_slice(df_input_slice):
    df_collect_list = _get_sample_spark_data_frame(df_input_slice)

    df_lookback = df_collect_list.select(
        "*",
        aggregate_over_slice(
            input_col="value_list",
            lower_bound=-3,
            upper_bound=-1,
            anchor="time_index",
            anchor_array="time_list",
            func=array_sum,
            alias="sum_val",
        ),
    )
    assert df_lookback.filter("time_index=1").select("sum_val").collect()[0][0] == 0
    assert df_lookback.filter("time_index=2").select("sum_val").collect()[0][0] == 0
    assert df_lookback.filter("time_index=3").select("sum_val").collect()[0][0] == 1

    df_lookfwd = df_collect_list.select(
        "*",
        aggregate_over_slice(
            input_col="value_list",
            lower_bound=1,
            upper_bound=3,
            anchor="time_index",
            anchor_array="time_list",
            func=array_sum,
            alias="sum_val",
        ),
    )
    assert df_lookfwd.filter("time_index=1").select("sum_val").collect()[0][0] == 6
    assert df_lookfwd.filter("time_index=10").select("sum_val").collect()[0][0] == 10
    assert df_lookfwd.filter("time_index=11").select("sum_val").collect()[0][0] == 0

    df_lookbtw = df_collect_list.select(
        "*",
        aggregate_over_slice(
            input_col="value_list",
            lower_bound=-1,
            upper_bound=3,
            anchor="time_index",
            anchor_array="time_list",
            func=array_sum,
            alias="sum_val",
        ),
    )
    assert df_lookbtw.filter("time_index=1").select("sum_val").collect()[0][0] == 6
    assert df_lookbtw.filter("time_index=10").select("sum_val").collect()[0][0] == 27
    assert df_lookbtw.filter("time_index=11").select("sum_val").collect()[0][0] == 19


def test_return_slice(df_input_slice):
    df = _get_sample_spark_data_frame(df_input_slice)

    result_even_even = df.select(
        "*",
        return_slice(
            lower_bound=0,
            upper_bound=0,
            anchor_array="time_list",
            value_array="value_list",
            anchor_col="time_index",
            alias="test",
        ),
    )

    collected_even_even = [x[0] for x in result_even_even.select("test").collect()]
    assert collected_even_even == [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
    ]

    result_negative_negative = df.select(
        "*",
        return_slice(
            lower_bound=-3,
            upper_bound=-2,
            anchor_array="time_list",
            value_array="value_list",
            anchor_col="time_index",
            alias="test",
        ),
    )

    collected_negative_negative = [
        x[0] for x in result_negative_negative.select("test").collect()
    ]
    assert collected_negative_negative == [
        [],
        [],
        [0],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
    ]

    result_negative_even = df.select(
        "*",
        return_slice(
            lower_bound=-3,
            upper_bound=0,
            anchor_array="time_list",
            value_array="value_list",
            anchor_col="time_index",
            alias="test",
        ),
    )

    collected_negative_even = [
        x[0] for x in result_negative_even.select("test").collect()
    ]
    assert collected_negative_even == [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
        [7, 8, 9, 10],
    ]

    result_even_positive = df.select(
        "*",
        return_slice(
            lower_bound=0,
            upper_bound=3,
            anchor_array="time_list",
            value_array="value_list",
            anchor_col="time_index",
            alias="test",
        ),
    )

    collected_even_positive = [
        x[0] for x in result_even_positive.select("test").collect()
    ]
    assert collected_even_positive == [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
        [7, 8, 9, 10],
        [8, 9, 10],
        [9, 10],
        [10],
    ]

    result_positive_positive = df.select(
        "*",
        return_slice(
            lower_bound=2,
            upper_bound=4,
            anchor_array="time_list",
            value_array="value_list",
            anchor_col="time_index",
            alias="test",
        ),
    )

    collected_positive_positive = [
        x[0] for x in result_positive_positive.select("test").collect()
    ]
    assert collected_positive_positive == [
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9, 10],
        [9, 10],
        [10],
        [],
        [],
    ]

    result_negative_positive = df.select(
        "*",
        return_slice(
            lower_bound=-2,
            upper_bound=3,
            anchor_array="time_list",
            value_array="value_list",
            anchor_col="time_index",
            alias="test",
        ),
    )

    collected_negative_positive = [
        x[0] for x in result_negative_positive.select("test").collect()
    ]
    assert collected_negative_positive == [
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 8, 9],
        [5, 6, 7, 8, 9, 10],
        [6, 7, 8, 9, 10],
        [7, 8, 9, 10],
        [8, 9, 10],
    ]
