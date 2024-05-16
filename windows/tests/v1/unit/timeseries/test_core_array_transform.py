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
from pyspark import Row
from pyspark.sql import functions as f

from feature_generation.v1.core.timeseries.array_transform import (
    array_derivative,
    array_distinct,
    array_local_peaks_from_derivative,
    array_smooth_ts_values,
    forward_fill,
    interpolate_constant,
    scipy_interpolate,
)


def test_array_distinct(mock_arrays_df):
    distinct = mock_arrays_df.select(
        array_distinct(value="value", alias="distinct")
    ).collect()[0][0]
    distinct.sort()
    assert distinct == [1, 2, 3]


def test_array_smooth_ts_values(mock_arrays_df):
    smooth_ts_values = mock_arrays_df.select(
        array_smooth_ts_values(value="value", length=2, alias="smooth_ts_values")
    ).collect()[0][0]
    assert smooth_ts_values == [1.5, 1.5, 2.0, 2.0, 0.5]


def test_array_derivative(mock_arrays_df):
    derivative = mock_arrays_df.select(
        array_derivative(value="value", time_delta=2, alias="derivative")
    ).collect()[0][0]
    assert derivative == [0.5, -0.5, 1.0, -1.0, 0.0]


def test_array_find_local_peaks(mock_arrays_df):
    local_peaks = (
        mock_arrays_df.select(
            "*", array_derivative(value="value", time_delta=1, alias="derivative")
        )
        .select(
            array_local_peaks_from_derivative(
                value="derivative", alias="local+peaks"
            ).alias("a"),
            "value",
        )
        .collect()[0][0]
    )
    assert local_peaks == [0, 1, -1, 1, -1]


def test_scipy_interpolate(mock_arrays_df):
    spine_df = mock_arrays_df.select(
        "*",
        f.array_max("time_index").cast("long").alias("max_time_index"),
        f.array_min("time_index").cast("long").alias("min_time_index"),
    ).select(
        "*",
        f.sequence(start="min_time_index", stop="max_time_index", step=f.lit(1)).alias(
            "spine_index"
        ),
    )

    interpolated_df = spine_df.select(
        "*",
        scipy_interpolate(
            "spine_index",
            time_index="time_index",
            value="value",
            kind="linear",
            alias="interpolated_val",
        ),
    )

    interpolated_values = interpolated_df.select("interpolated_val").collect()[0][0]

    assert interpolated_values == [1.0, 2.0, 1.0, 3.0, 2.0, 1.0]


def test_forward_fill(mock_arrays_df):
    spine_df = mock_arrays_df.select(
        "*",
        f.array_max("time_index").cast("long").alias("max_time_index"),
        f.array_min("time_index").cast("long").alias("min_time_index"),
    ).select(
        "*",
        f.sequence(start="min_time_index", stop="max_time_index", step=f.lit(1)).alias(
            "spine_index"
        ),
    )

    interpolated_df = spine_df.select(
        "*",
        forward_fill("spine_index", "time_index", "value", alias="interpolated_val"),
    )

    interpolated_values = interpolated_df.select("interpolated_val").collect()[0][0]

    assert interpolated_values == [1.0, 2.0, 1.0, 3.0, 3.0, 1.0]


@pytest.fixture
def df_input_interpolate_null(spark):
    return spark.createDataFrame(
        [
            Row(
                npi_id="h1",
                time_index=1,
                value=0,
                spine_index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                time_index_array=[1, 4, 6, 7, 8, 11],
                value_array=[0, 3, 5, 6, 7, 8, 11],
            ),
            Row(
                npi_id="h1",
                time_index=4,
                value=3,
                spine_index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                time_index_array=[1, 4, 6, 7, 8, 11],
                value_array=[0, 3, 5, 6, 7, 8, 11],
            ),
            Row(
                npi_id="h1",
                time_index=6,
                value=5,
                spine_index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                time_index_array=[1, 4, 6, 7, 8, 11],
                value_array=[0, 3, 5, 6, 7, 8, 11],
            ),
            Row(
                npi_id="h1",
                time_index=7,
                value=6,
                spine_index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                time_index_array=[1, 4, 6, 7, 8, 11],
                value_array=[0, 3, 5, 6, 7, 8, 11],
            ),
            Row(
                npi_id="h1",
                time_index=8,
                value=7,
                spine_index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                time_index_array=[1, 4, 6, 7, 8, 11],
                value_array=[0, 3, 5, 6, 7, 8, 11],
            ),
            Row(
                npi_id="h1",
                time_index=11,
                value=10,
                spine_index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                time_index_array=[1, 4, 6, 7, 8, 11],
                value_array=[0, 3, 5, 6, 7, 8, 11],
            ),
        ]
    )


def test_interpolate_null(df_input_interpolate_null):
    list_of_tags = ["value"]
    df_with_interpolated_null = df_input_interpolate_null.select(
        "*",
        *[
            interpolate_constant(
                spine_index="spine_index",
                time_index="time_index_array",
                value=f"{x}_array",
                alias=f"{x}_array_padded",
            )
            for x in list_of_tags
        ],
    )

    assert (
        df_with_interpolated_null.select(f.size("value_array_padded")).collect()[0][0]
        == df_with_interpolated_null.select(f.size("spine_index")).collect()[0][0]
    )
    assert (
        df_with_interpolated_null.select(f.size("value_array_padded")).collect()[0][0]
        > df_with_interpolated_null.select(f.size("time_index_array")).collect()[0][0]
    )
    assert df_with_interpolated_null.select("value_array_padded").collect()[0][0] == [
        0.0,
        None,
        None,
        3.0,
        None,
        5.0,
        6.0,
        7.0,
        None,
        None,
        8.0,
    ]


def test_interpolate_value(df_input_interpolate_null):
    list_of_tags = ["value"]
    df_with_interpolated_null = df_input_interpolate_null.select(
        "*",
        *[
            interpolate_constant(
                spine_index="spine_index",
                time_index="time_index_array",
                value=f"{x}_array",
                alias=f"{x}_array_padded",
                constant=0,
            )
            for x in list_of_tags
        ],
    )

    assert (
        df_with_interpolated_null.select(f.size("value_array_padded")).collect()[0][0]
        == df_with_interpolated_null.select(f.size("spine_index")).collect()[0][0]
    )
    assert (
        df_with_interpolated_null.select(f.size("value_array_padded")).collect()[0][0]
        > df_with_interpolated_null.select(f.size("time_index_array")).collect()[0][0]
    )
    assert df_with_interpolated_null.select("value_array_padded").collect()[0][0] == [
        0.0,
        0.0,
        0.0,
        3.0,
        0.0,
        5.0,
        6.0,
        7.0,
        0.0,
        0.0,
        8.0,
    ]
