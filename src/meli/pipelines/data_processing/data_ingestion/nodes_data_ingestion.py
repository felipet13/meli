"""Wrapper for processor class and functions found in processor module.

Serves as entry point.
"""

from typing import Dict, Union

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame

from .core.processor import Processor


def ingest_historical_data(
    raw_df: Union[SparkDataFrame, pd.DataFrame],
    input_parameters_dict: Dict,
) -> SparkDataFrame:
    """Return the sliced data for historical ingestion.

    Args:
        raw_df: Input dataframe for historical processing.
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
    raw_df: Union[SparkDataFrame, pd.DataFrame],
    input_parameters_dict: Dict,
    existing_data_df: Union[SparkDataFrame, pd.DataFrame],
) -> SparkDataFrame:
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
