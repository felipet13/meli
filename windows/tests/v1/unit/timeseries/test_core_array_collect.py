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

import datetime

import pyspark.sql.functions as f
import pytest
from pyspark import Row
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from feature_generation.v1.core.timeseries.array_collect import (
    collect_array_then_interpolate,
    create_spine_array_from_index_array,
    sorted_collect_list,
)
from feature_generation.v1.core.timeseries.array_transform import (
    interpolate_constant,
    scipy_interpolate,
)
from feature_generation.v1.core.timeseries.datetime import minute_index


@pytest.fixture
def mock_datetime_df(spark):
    """Set timezone for consistent behaviour across machines.

    Spark takes into account system time zone.
    """
    schema = StructType(
        [
            StructField("element_id", StringType(), True),
            StructField("reading_ts", TimestampType(), True),
            StructField("reading_val", IntegerType(), True),
            StructField("reading_val2", IntegerType(), True),
        ]
    )

    data = [
        (
            "line1",
            datetime.datetime(1970, 1, 1, 5, 0, tzinfo=datetime.timezone.utc),
            1,
            2,
        ),
        (
            "line1",
            datetime.datetime(1970, 1, 1, 5, 2, tzinfo=datetime.timezone.utc),
            2,
            3,
        ),
        (
            "line1",
            datetime.datetime(1970, 1, 1, 5, 6, tzinfo=datetime.timezone.utc),
            3,
            4,
        ),
        (
            "line2",
            datetime.datetime(1970, 1, 1, 7, 1, tzinfo=datetime.timezone.utc),
            3,
            5,
        ),
        (
            "line2",
            datetime.datetime(1970, 1, 1, 7, 5, tzinfo=datetime.timezone.utc),
            1,
            6,
        ),
        (
            "line2",
            datetime.datetime(1970, 1, 1, 7, 10, tzinfo=datetime.timezone.utc),
            5,
            7,
        ),
    ]

    mock_datetime_df = spark.createDataFrame(data, schema)

    return mock_datetime_df


def test_collect_array_then_interpolate(mock_datetime_df, spark):
    df_with_index = mock_datetime_df.withColumn(
        "minute_index", minute_index("reading_ts")
    )

    collected_array_df = collect_array_then_interpolate(
        df=df_with_index,
        order="minute_index",
        values=["reading_val", "reading_val2"],
        groupby="element_id",
        interpolate_func=interpolate_constant,
        desc=True,
    )

    line_1_array = (
        collected_array_df.filter(f.col("element_id") == "line1")
        .select("reading_val")
        .collect()[0][0]
    )

    line_2_array_val2 = (
        collected_array_df.filter(f.col("element_id") == "line2")
        .select("reading_val2")
        .collect()[0][0]
    )

    assert line_1_array == [3, 2, 1]
    assert line_2_array_val2 == [7, 6, 5]


def test_spine_collect_array_then_interpolate(mock_datetime_df, spark):
    df_with_index = mock_datetime_df.withColumn(
        "minute_index", minute_index("reading_ts")
    )

    collected_array_df = collect_array_then_interpolate(
        df=df_with_index,
        order="minute_index",
        values=["reading_val", "reading_val2"],
        groupby="element_id",
        spine="index_spine",
        interpolate_func=interpolate_constant,
    )

    line_2_index_spine_array = (
        collected_array_df.filter(f.col("element_id") == "line2")
        .select("index_spine")
        .collect()[0][0]
    )

    assert line_2_index_spine_array == [
        421,
        422,
        423,
        424,
        425,
        426,
        427,
        428,
        429,
        430,
    ]
    assert set(collected_array_df.columns) == set(
        [
            "reading_val2",
            "minute_index",
            "reading_val",
            "element_id",
            "index_spine",
            "reading_val_array_padded",
            "reading_val2_array_padded",
        ]
    )


def test_spine_collect_array_then_interpolate_null_padded(mock_datetime_df, spark):
    df_with_index = mock_datetime_df.withColumn(
        "minute_index", minute_index("reading_ts")
    )

    collected_array_df = collect_array_then_interpolate(
        df=df_with_index,
        order="minute_index",
        values=["reading_val", "reading_val2"],
        groupby="element_id",
        spine="index_spine",
        interpolate_func=interpolate_constant,
    )

    line1_reading_val_array_padded = (
        collected_array_df.filter(f.col("element_id") == "line1")
        .select("reading_val_array_padded")
        .collect()[0][0]
    )

    assert line1_reading_val_array_padded == [1.0, None, 2.0, None, None, None, 3.0]
    assert set(collected_array_df.columns) == set(
        [
            "element_id",
            "reading_val",
            "reading_val2",
            "minute_index",
            "index_spine",
            "reading_val_array_padded",
            "reading_val2_array_padded",
        ]
    )


def test_spine_collect_array_then_interpolate_num_padded(mock_datetime_df, spark):
    df_with_index = mock_datetime_df.withColumn(
        "minute_index", minute_index("reading_ts")
    )
    collected_array_df = collect_array_then_interpolate(
        df=df_with_index,
        order="minute_index",
        values=["reading_val", "reading_val2"],
        groupby="element_id",
        spine="index_spine",
        interpolate_func=interpolate_constant,
        constant=0,
    )

    line1_reading_val_array_padded = (
        collected_array_df.filter(f.col("element_id") == "line1")
        .select("reading_val_array_padded")
        .collect()[0][0]
    )

    assert line1_reading_val_array_padded == [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    assert set(collected_array_df.columns) == set(
        [
            "element_id",
            "reading_val",
            "reading_val2",
            "minute_index",
            "index_spine",
            "reading_val_array_padded",
            "reading_val2_array_padded",
        ]
    )


def test_spine_collect_array_then_interpolate_scipy_interpolate(
    mock_datetime_df, spark
):
    df_with_index = mock_datetime_df.withColumn(
        "minute_index", minute_index("reading_ts")
    )
    collected_array_df = collect_array_then_interpolate(
        df=df_with_index,
        order="minute_index",
        values=["reading_val", "reading_val2"],
        groupby="element_id",
        spine="index_spine",
        interpolate_func=scipy_interpolate,
        kind="previous",
    )

    line1_reading_val_array_padded = (
        collected_array_df.filter(f.col("element_id") == "line1")
        .select("reading_val_array_padded")
        .collect()[0][0]
    )

    assert line1_reading_val_array_padded == [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]
    assert set(collected_array_df.columns) == set(
        [
            "element_id",
            "reading_val",
            "reading_val2",
            "minute_index",
            "index_spine",
            "reading_val_array_padded",
            "reading_val2_array_padded",
        ]
    )


@pytest.fixture
def df_input_spine_index(spark):
    return spark.createDataFrame(
        [
            Row(
                npi_id="h1",
                time_index=[1, 4, 6, 7, 8, 11],
                value=[0, 3, 5, 6, 7, 8, 11],
            ),
            Row(
                npi_id="h2",
                time_index=[1, 4, 6, 7, 8, 11],
                value=[0, 3, 5, 6, 7, 8, 11],
            ),
        ]
    )


def test_create_spine_array_from_index_array(df_input_spine_index):
    results = create_spine_array_from_index_array(
        df=df_input_spine_index, order_array="time_index", alias="spine"
    )
    assert results.select("spine").distinct().collect()[0][0] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]


def test_sorted_collect_list(mock_before_arrays):
    results = (
        mock_before_arrays.groupBy("person")
        .agg(sorted_collect_list("time_index", "value", alias="sorted_collect_list"))
        .collect()
    )

    sorted_collect = [x["time_index"] for x in results[0]["sorted_collect_list"]]
    assert sorted_collect == [1, 2, 3, 4, 5]


def test_sorted_collect_list_desc(mock_before_arrays):
    results = (
        mock_before_arrays.groupBy("person")
        .agg(
            sorted_collect_list(
                "time_index", "value", True, alias="sorted_collect_list"
            )
        )
        .collect()
    )

    sorted_collect = [x["time_index"] for x in results[0]["sorted_collect_list"]]
    assert sorted_collect == [5, 4, 3, 2, 1]
