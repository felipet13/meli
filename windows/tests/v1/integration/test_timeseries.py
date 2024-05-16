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
import datetime

import pytest
from pyspark.sql import functions as f
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from feature_generation.v1.core.timeseries.array_aggregate import array_auc_time_delta
from feature_generation.v1.core.timeseries.array_collect import (
    collect_array_then_interpolate,
)
from feature_generation.v1.core.timeseries.array_transform import (
    array_derivative,
    array_local_peaks_from_derivative,
    array_smooth_ts_values,
    interpolate_constant,
    scipy_interpolate,
)


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
        ("line1", datetime.datetime(1970, 1, 1, 5, 0), 1),
        ("line1", datetime.datetime(1970, 1, 1, 5, 30), 5),
        ("line1", datetime.datetime(1970, 1, 1, 6, 0), 2),
        ("line1", datetime.datetime(1970, 1, 1, 7, 30), 3),
        ("line2", datetime.datetime(1970, 1, 1, 7, 33), 8),
        ("line2", datetime.datetime(1970, 1, 1, 8, 0), 4),
        ("line2", datetime.datetime(1970, 1, 1, 9, 30), 5),
    ]

    return spark.createDataFrame(data, schema)


def test_e2e_ts_array(spark, mock_timeseries_df):
    spine_df = collect_array_then_interpolate(
        df=mock_timeseries_df.withColumn(
            "index_start_ts", f.col("start_ts").cast("long")
        ),
        groupby=["element_id"],
        order="index_start_ts",
        values=["reading_val"],
        spine="spine_index",
        delta=30 * 60,
        interpolate_func=interpolate_constant,
    )

    interpolated_df = spine_df.select(
        f.lit(30).alias("time_delta"),
        "element_id",
        scipy_interpolate(
            "spine_index",
            "index_start_ts",
            "reading_val",
            "previous",
            alias="interpolated_reading_val",
        ),
    )

    smooth_df = interpolated_df.withColumn("time_delta", f.lit(30)).select(
        "*",
        array_smooth_ts_values(
            value="interpolated_reading_val", length=2, alias="smooth_val"
        ),
    )
    auc_df = smooth_df.select(
        "*",
        array_auc_time_delta(
            value="interpolated_reading_val", time_delta="time_delta", alias="auc"
        ),
    )

    deriv_df = auc_df.select(
        "*",
        array_derivative(
            value="smooth_val", time_delta="time_delta", alias="fderiv_val"
        ),
    )

    peaks_df = deriv_df.select(
        "*", array_local_peaks_from_derivative(value="fderiv_val", alias="peaks")
    )

    peaks = peaks_df.select("peaks").filter("element_id == 'line1'").collect()[0][0]
    auc = peaks_df.select("auc").filter("element_id == 'line1'").collect()[0][0]

    assert peaks == [0, 1, -1, 0, 1, -1]
    assert auc == 390.0
