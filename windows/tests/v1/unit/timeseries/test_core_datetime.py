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
from pyspark.sql.types import (
    DateType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from feature_generation.v1.core.timeseries.datetime import (
    date_index,
    hour_index,
    minute_index,
    month_index,
    quarter_index,
    time_add,
)


@pytest.fixture
def mock_date_df(spark):
    data = [
        ["bla-1", datetime.date(2019, 1, 1)],
        ["bla-2", datetime.date(2019, 2, 1)],
        ["bla-3", datetime.date(2019, 3, 1)],
        ["bla-1", datetime.date(2019, 4, 1)],
        ["bla-2", datetime.date(2019, 5, 1)],
        ["bla-3", datetime.date(2019, 6, 1)],
    ]

    schema = StructType(
        [
            StructField("element_id", StringType()),
            StructField("observation_dt", DateType()),
        ]
    )

    df = spark.createDataFrame(data, schema)

    return df


@pytest.fixture
def mock_timestamp_df(spark):
    schema = StructType(
        [
            StructField("element_id", StringType(), True),
            StructField("start_ts", TimestampType(), True),
        ]
    )

    data = [
        ("line1", datetime.datetime(1970, 1, 1, 5, 0)),
        ("line1", datetime.datetime(1970, 1, 1, 6, 0)),
        ("line1", datetime.datetime(1970, 1, 1, 7, 30)),
        ("line2", datetime.datetime(1970, 1, 1, 8, 0)),
        ("line2", datetime.datetime(1970, 1, 1, 9, 30)),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def mock_timeseries_df(spark):
    schema = StructType(
        [
            StructField("element_id", StringType(), True),
            StructField("start_ts", TimestampType(), True),
            StructField("reading_val", IntegerType(), True),
        ]
    )

    data = [
        ("line1", datetime.datetime(1970, 1, 1, 5, 0, tzinfo=datetime.timezone.utc), 1),
        ("line1", datetime.datetime(1970, 1, 1, 6, 0, tzinfo=datetime.timezone.utc), 2),
        (
            "line1",
            datetime.datetime(1970, 1, 1, 7, 30, tzinfo=datetime.timezone.utc),
            3,
        ),
        (
            "line2",
            datetime.datetime(1970, 1, 1, 7, 33, tzinfo=datetime.timezone.utc),
            3,
        ),
        ("line2", datetime.datetime(1970, 1, 1, 8, 0, tzinfo=datetime.timezone.utc), 4),
        (
            "line2",
            datetime.datetime(1970, 1, 1, 9, 30, tzinfo=datetime.timezone.utc),
            5,
        ),
    ]

    return spark.createDataFrame(data, schema)


def test_date_index(_get_sample_spark_data_frame2):
    df = _get_sample_spark_data_frame2

    sum_diff = (
        df.withColumn("new_date_index", date_index("date"))
        .withColumn("diff", f.col("date_index") - f.col("new_date_index"))
        .select(f.sum("diff"))
    ).collect()[0][0]

    assert sum_diff == 0


def test_date_index_reference(_get_sample_spark_data_frame2):
    df = _get_sample_spark_data_frame2

    min_and_max = (
        df.withColumn("new_date_index", date_index("date", "2012-05-01"))
        .select(
            f.min(f.col("new_date_index")).alias("min"),
            f.max(f.col("new_date_index")).alias("max"),
        )
        .collect()[0]
    )
    assert min_and_max.min == 0
    assert min_and_max.max == 14


def test_bad_reference_date(_get_sample_spark_data_frame2):
    df = _get_sample_spark_data_frame2

    with pytest.raises(ValueError, match=r"Date format should follow yyyy-mm-dd."):
        df.withColumn("new_date_index", date_index("date", "01-01-1970"))


def test_month_index(mock_date_df):
    df = mock_date_df

    month_indices = [
        x[0]
        for x in (
            df.withColumn("month_index", month_index("observation_dt")).select(
                "month_index"
            )
        ).collect()
    ]

    assert month_indices == [588, 589, 590, 591, 592, 593]


def test_hour_index(mock_timestamp_df):
    df = mock_timestamp_df

    hour_indices = [
        x[0]
        for x in (
            df.withColumn("hour_index", hour_index("start_ts")).select("hour_index")
        ).collect()
    ]

    assert hour_indices == [5, 6, 7.5, 8, 9.5]


def test_hour_index_reference(mock_timestamp_df):
    df = mock_timestamp_df

    hour_indices = [
        x[0]
        for x in (
            df.withColumn(
                "hour_index", hour_index("start_ts", "1970-01-01 05:00:00")
            ).select("hour_index")
        ).collect()
    ]

    assert hour_indices == [0.0, 1.0, 2.5, 3.0, 4.5]


def test_minute_index_reference(mock_timestamp_df):
    df = mock_timestamp_df

    minute_indices = [
        x[0]
        for x in (
            df.withColumn(
                "hour_index", minute_index("start_ts", "1970-01-01 05:00:00")
            ).select("hour_index")
        ).collect()
    ]

    assert minute_indices == [0, 60, 150, 180, 270]


def test_time_add(_get_sample_spark_data_frame2):
    df = _get_sample_spark_data_frame2.withColumn(
        "date", f.col("date").cast("timestamp")
    )
    result_df = df.withColumn("time_plus_10_mins", time_add("date", "10 minutes"))
    gendry_value = (
        result_df.filter(f.col("name") == "Gendry")
        .filter(f.col("date_index") == 15461)
        .select(f.col("time_plus_10_mins").cast("string"))
        .distinct()
        .collect()[0][0]
    )
    assert gendry_value == "2012-05-01 00:10:00"


def test_time_add_days(_get_sample_spark_data_frame2):
    df = _get_sample_spark_data_frame2.withColumn(
        "date", f.col("date").cast("timestamp")
    )
    result_df = df.withColumn("time_minus_1_day", time_add("date", "-1 days"))
    gendry_value = (
        result_df.filter(f.col("name") == "Gendry")
        .filter(f.col("date_index") == 15461)
        .select(f.col("time_minus_1_day").cast("string"))
        .distinct()
        .collect()[0][0]
    )
    assert gendry_value == "2012-04-30 00:00:00"


def test_time_add_date(_get_sample_spark_data_frame2):
    result_df = _get_sample_spark_data_frame2.withColumn(
        "time_minus_1_day", time_add("date", "-1 days")
    )
    gendry_value = (
        result_df.filter(f.col("name") == "Gendry")
        .filter(f.col("date_index") == 15461)
        .select(f.col("time_minus_1_day").cast("string"))
        .distinct()
        .collect()[0][0]
    )
    assert gendry_value == "2012-04-30"


def test_quarter_index(mock_date_df):
    result_df = mock_date_df.withColumn(
        "quarter_index", quarter_index("observation_dt")
    )

    assert [x[0] for x in result_df.select("quarter_index").collect()] == [
        196.0,
        196.33333333333334,
        196.66666666666666,
        197.0,
        197.33333333333334,
        197.66666666666666,
    ]
