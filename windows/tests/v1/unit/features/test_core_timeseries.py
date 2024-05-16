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

"""Test functions to produce custom column features."""

# pylint: skip-file
# flake8: noqa

import pytest
from pyspark.sql import functions as f

from feature_generation.v1.core.features.create_column import create_columns_from_config
from feature_generation.v1.core.features.timeseries import (
    abs_relative_variation,
    add_aggregate_value_column,
    auc_trapezoidal,
    find_local_peaks,
    weighted_avg,
)


@pytest.fixture(scope="module")
def auc_test_column_instructions():
    column_instructions = auc_trapezoidal(
        y="x_flag",
        x="date",
        partition_by="name",
    )

    return column_instructions


@pytest.fixture(scope="module")
def wt_avg_test_column_instructions():
    column_instructions = weighted_avg(
        y="x_flag",
        x="date",
        partition_by="name",
        alias="test_wt_avg",
    )

    return column_instructions


def test_auc_trapezoidal_values(_get_sample_spark_data_frame2):
    df = _get_sample_spark_data_frame2.withColumn(
        "date", f.col("date").cast("timestamp")
    )

    result_df = df.withColumn(
        "test_area",
        auc_trapezoidal(y="x_flag", x="date", partition_by="name"),
    )

    gendry_auc = (
        result_df.filter(f.col("name") == "Gendry")
        .groupby("name")
        .agg(f.sum("test_area").alias("area"))
        .select("area")
    ).collect()[0][0]
    assert gendry_auc == 172800.0

    cersei_auc = (
        result_df.filter(f.col("name") == "Cersei")
        .groupby("name")
        .agg(f.sum("test_area").alias("area"))
        .select("area")
    ).collect()[0][0]
    assert cersei_auc == 302400.0


@pytest.mark.xfail(raises=TypeError)
def test_auc_trapezoidal_wrong_config(_get_sample_spark_data_frame2):
    column_instructions = weighted_avg(y="x_flag", x="date", partition_by="name")

    create_columns_from_config(
        df=_get_sample_spark_data_frame2, column_instructions=column_instructions
    )


def test_weighted_avg_column_created(
    _get_sample_spark_data_frame2, wt_avg_test_column_instructions
):
    add_columns_conf = [wt_avg_test_column_instructions]
    df = _get_sample_spark_data_frame2.withColumn(
        "date", f.col("date").cast("timestamp")
    )

    result_df = create_columns_from_config(df, add_columns_conf)
    assert "test_wt_avg" in result_df.columns


def test_weighted_avg_values(_get_sample_spark_data_frame2):
    df = _get_sample_spark_data_frame2.withColumn(
        "date", f.col("date").cast("timestamp")
    )

    result_df = (
        df.withColumn("test_ts", f.col("date").cast("timestamp"))
        .withColumn(
            "wa_non_ts",
            weighted_avg(
                partition_by="name",
                y="date_index",
                x="x_flag",
                is_weight_sequence=False,
            ),
        )
        .withColumn(
            "wa_time_index",
            weighted_avg(
                partition_by="name",
                y="x_flag",
                x="date_index",
                is_weight_sequence=True,
            ),
        )
        .withColumn(
            "wa_time_series",
            weighted_avg(
                partition_by="name",
                y="x_flag",
                x="test_ts",
                is_weight_sequence=True,
            ),
        )
    )

    gendry_wa = (
        result_df.filter(f.col("name") == "Gendry")
        .groupby("name")
        .agg(f.max("wa_non_ts").alias("wa_non_ts"))
        .select("wa_non_ts")
    ).collect()[0][0]
    assert gendry_wa == 15463.0

    cersei_wa = (
        result_df.filter(f.col("name") == "Cersei")
        .groupby("name")
        .agg(f.max("wa_time_index").alias("wa_time_index"))
        .select("wa_time_index")
    ).collect()[0][0]
    assert cersei_wa == 0.8

    arya_wa = (
        result_df.filter(f.col("name") == "Arya")
        .groupby("name")
        .agg(f.max("wa_time_series").alias("wa_time_series"))
        .select("wa_time_series")
    ).collect()[0][0]
    assert arya_wa == 0.5


def test_abs_relative_variation(get_sample_spark_data_frame):
    df = get_sample_spark_data_frame

    df = df.withColumn("variation", abs_relative_variation(f.col("age"), f.lit(25)))

    for age, variation in df.select("age", "variation").collect():
        assert (age, variation) == (age, abs(round(1 - age / 25, 2)))


def test_add_aggregate_value_column(_get_sample_spark_data_frame2):
    result_df = _get_sample_spark_data_frame2.withColumn(
        "min_date_index",
        add_aggregate_value_column("name", f.min("date_index")),
    )
    gendry_value = (
        result_df.filter(f.col("name") == "Arya")
        .select("min_date_index")
        .distinct()
        .collect()[0][0]
    )
    assert gendry_value == 15466


def test_local_peaks(mock_tag_df):
    df = mock_tag_df.withColumn("date", f.col("date").cast("timestamp"))
    result_df = df.withColumn("time_col_long", f.col("date").cast("long")).withColumn(
        "peaks", find_local_peaks("name", "time_col_long", "val", 60, 0.02)
    )
    result_df.cache()
    gendry_df = (
        result_df.filter(
            (f.col("name") == "batch_0001") & (f.col("date_index") == 1)
        ).select(f.col("peaks"))
    ).collect()[0][0]
    assert gendry_df == "maxima_1"
