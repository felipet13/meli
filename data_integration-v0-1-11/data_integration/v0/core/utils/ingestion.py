# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.

"""Main functions to select data slides based on given time ranges."""

import pyspark
import pyspark.sql.functions as f

from data_integration.v0.core.utils.date_validation import validate_date


def get_only_incremental_data(
    input_df: pyspark.sql.DataFrame,
    existing_data_df: pyspark.sql.DataFrame,
    date_col_name: str,
) -> pyspark.sql.DataFrame:
    """Filter all data since the last offset.

    Args:
        input_df: Raw data.
        existing_data_df: Existing dataframe.
        date_col_name: Date column name to be given.

    Returns:
        DataFrame: output data
    """
    # select most recent date for which data is already processed
    max_date = existing_data_df.select(
        f.max(f.col(date_col_name)).alias("offset_dt")
    ).first()[0]

    # create new column with this value
    incremental_filter_df = input_df.withColumn(
        "offset_dt", f.to_date(f.lit(max_date.strftime(r"%Y-%m-%d")))
    )
    # use it for filter
    incremental_filter_df = incremental_filter_df.filter(
        f.col(date_col_name) > f.col("offset_dt")
    ).drop("offset_dt")

    return incremental_filter_df


def get_only_filtered_data(
    df: pyspark.sql.DataFrame,
    partition_col: str,
    start_dt: str = None,
    end_dt: str = None,
) -> pyspark.sql.DataFrame:
    """Filter data based on a date range.

    Args:
        df: Data to be filtered.
        partition_col: Column to filter by.
        start_dt: Initial filter date in format "YYYY-MM-dd".
        end_dt: End filter date in format "YYYY-MM-dd".

    Returns:
        DataFrame: output data.
    """
    start_dt = None if str(start_dt).lower() == "none" else start_dt
    end_dt = None if str(end_dt).lower() == "none" else end_dt

    if start_dt and end_dt:
        validate_date([start_dt, end_dt])
        df = df.filter(f.col(partition_col).between(start_dt, end_dt))

    return df
