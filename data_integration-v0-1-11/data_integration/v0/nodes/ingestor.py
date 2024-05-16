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
"""Wrapper for processor class and functions found in processor module.

Serves as entry point in hydra.
"""

from typing import Dict, Union

import pandas as pd
import pyspark

from data_integration.v0.core.processor import Processor


def transform_data(
    raw_df: Union[pyspark.sql.DataFrame, pd.DataFrame],
    input_parameters_dict: Dict,
    log_spark_info: bool = False,
) -> pyspark.sql.DataFrame:
    """Apply data transformations on the input dataframe.

    Args:
        raw_df: Input dataframe to apply data transformation.
        input_parameters_dict: Dict of params to be passed to the processor class
        constructor.
        log_spark_info: Logs the final Spark plan in `DEBUG` level.

    Returns:
        Dataframe already transformed.
    """
    # Instantiation
    processor = Processor()

    # Produce transformed data
    historical_data = processor.transform(raw_df, input_parameters_dict, log_spark_info)
    return historical_data


def ingest_historical_data(
    raw_df: Union[pyspark.sql.DataFrame, pd.DataFrame],
    input_parameters_dict: Dict,
) -> pyspark.sql.DataFrame:
    """Return the sliced data for incremental ingestion.

    Args:
        raw_df: Input dataframe for incremental processing.
        input_parameters_dict: Dict of params to be passed to the processor class
            constructor.

    Returns:
        Dataframe containing historical data slice to save.
    """
    # Instantiation
    processor = Processor()

    # Produce sliced data
    historical_data = processor.ingest_historical_data(
        input_parameters_dict=input_parameters_dict,
        raw_df=raw_df,
    )
    return historical_data


def ingest_incremental_data(
    raw_df: Union[pyspark.sql.DataFrame, pd.DataFrame],
    input_parameters_dict: Dict,
    existing_data_df: Union[pyspark.sql.DataFrame, pd.DataFrame],
) -> pyspark.sql.DataFrame:
    """Return the sliced data for incremental ingestion.

    Args:
        raw_df: Input dataframe for incremental processing.
        input_parameters_dict: Dict of params to be passed to the processor class
            constructor.
        existing_data_df: Dataframe that contains the already existing data,
            that can be used to identify which is the new data on `raw_df`.

    Returns:
        Dataframe containing incremental data to save.
    """
    # Instantiation
    processor = Processor()

    # Produce sliced data
    incremental_data = processor.ingest_incremental_data(
        raw_df=raw_df,
        input_parameters_dict=input_parameters_dict,
        existing_data_df=existing_data_df,
    )
    return incremental_data
