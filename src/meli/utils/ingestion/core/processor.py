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
# pylint: disable=invalid-name
"""Data-Ingestion processor module."""

import logging
from pprint import pformat
from typing import Dict, Union

import pandas as pd
import pyspark
from pyspark.sql import functions as f

from ingestion.core.utils.ingestion import (
    get_only_filtered_data,
    get_only_incremental_data,
)


class Processor:
    """Process data ingestion according to received parameters.

    The main entry point of the class is the method `ingest_data`.
    """

    def __init__(self) -> None:
        """__init__ method."""
        self.df = None
        self.parameters_dict = None
        self.logger = logging.getLogger(__name__)

    def ingest_incremental_data(
        self,
        raw_df: Union[pyspark.sql.DataFrame, pd.DataFrame],
        input_parameters_dict: Dict,
        existing_data_df: Union[pyspark.sql.DataFrame, pd.DataFrame],
    ) -> pyspark.sql.DataFrame:
        """Process the integration steps, according to parameterized options.

        Args:
            raw_df: Input dataframe for full processing (Historical).

            input_parameters_dict: Key/value parameters to the integration pipeline.
                This dictionary can contain the following keys:

                - `date_col_by_expr` [Dict] - [Optional]: Use Spark SQL expression to
                create a new date column or change name to existing date column. Pass
                multiple SQL expresion if intended to create more than 1 column.

                    {"new_col_name1": "SQL_EXPR1",
                    "new_col_name2": "SQL_EXPR2"}

                - `col_to_filter_by`: Name of column used to filter dataframe rows.

            existing_data_df: Dataframe that contains the already existing data,
            that can be used to identify which is the new data on `raw_df`.

        Raises:
            ValueError: When inconsistent or a missing parameters state are found during
                processing. Log messages will clarify the reason of the exception.

        Returns:
            A dataframe already filtered and partitioned, according to the parameters
                present in `input_parameters_dict`.
        """
        # if raw_df is not a Spark DFs, try to convert it from Pandas
        df_name = "raw_df"
        # passing df_name is necessary to generate meaningful output messages
        self.df = self._convert_dataframe(df_name, raw_df)
        self.logger.info("{df_name} schema:\n%s", pformat(self.df.schema))

        # validate the input parameters dictionary
        self.parameters_dict = self._validate_parameters_dict(
            input_parameters_dict, historical=True
        )

        # try to convert filter column to date type if it is not of type already
        self.df = self._convert_column_to_date(
            self.df, self.parameters_dict["col_to_filter_by"]
        )

        # get only the new data that still does not exist on `existing_data_df`
        # uses a date column to check what is new
        self.df = get_only_incremental_data(
            input_df=self.df,
            existing_data_df=existing_data_df,
            date_col_name=self.parameters_dict["col_to_filter_by"],
        )

        # Finally returns the incremental DataFrame
        return self.df

    def ingest_historical_data(
        self,
        raw_df: Union[pyspark.sql.DataFrame, pd.DataFrame],
        input_parameters_dict: Dict,
    ) -> pyspark.sql.DataFrame:
        """Process the integration steps, according to parameterized options.

        Args:
            raw_df: Input dataframe for full processing (Historical).

            input_parameters_dict: Key/value parameters to the integration pipeline.
                This dictionary can contain the following keys:

                - `col_to_filter_by`: Name of column to be used to filter dataframe rows
                - `start_dt` [Optional]: Sets the initial date that `raw_df` dataframe
                should be filtered.
                - `end_dt` [Optional]: Sets the ending date that `raw_df` dataframe
                should be filtered.

        Raises:
            ValueError: When inconsistent or a missing parameters state are found during
                processing. Log messages will clarify the reason of the exception.

        Returns:
            A dataframe already filtered and partitioned, according to the parameters
                present in `input_parameters_dict`.
        """
        # if raw_df is not a Spark DFs, try to convert it from Pandas
        df_name = "raw_df"
        # passing df_name is necessary to generate meaningful output messages
        self.df = self._convert_dataframe(df_name, raw_df)
        self.logger.info("{df_name} schema:\n%s", pformat(self.df.schema))

        # validate the input parameters dictionary
        self.parameters_dict = self._validate_parameters_dict(
            input_parameters_dict, historical=True
        )

        # try to convert filter column to date type if it is not of type already
        self.df = self._convert_column_to_date(
            self.df, self.parameters_dict["col_to_filter_by"]
        )

        # [Optional] filter a range of dates
        if self.parameters_dict.get("start_dt", None) and self.parameters_dict.get(
            "end_dt", None
        ):
            self.df = get_only_filtered_data(
                df=self.df,
                partition_col=self.parameters_dict["col_to_filter_by"],
                start_dt=self.parameters_dict["start_dt"],
                end_dt=self.parameters_dict["end_dt"],
            )

        # Finally returns the filtered DF
        return self.df

    @staticmethod
    def _convert_dataframe(
        df_name: str, input_df: Union[pd.DataFrame, pyspark.sql.DataFrame]
    ) -> pyspark.sql.DataFrame:
        """Converts a pandas.Dataframe to pyspark.sql.DataFrame, if necessary.

        Args:
            df_name: The name of the dataframe being casted, to be used in logging
             or Exception outputs.
            input_df: A pyspark.sql.DataFrame or pandas.Dataframe.

        Raises:
            ValueError: If the dataframe `input_df` is not a pandas.Dataframe or a
             pyspark.sql.DataFrame.

        Returns:
            A converted PySpark Dataframe
        """
        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        try:
            input_df = (
                input_df
                if isinstance(input_df, pyspark.sql.DataFrame)
                else spark.createDataFrame(input_df)
            )
        except ValueError as exc:
            raise ValueError(
                f"Expected Spark or Pandas dataframe in `{df_name}`, "
                f"got {type(input_df)}"
            ) from exc

        return input_df

    def _validate_parameters_dict(  # noqa: C901
        self, original_parameters: Dict, historical: bool
    ) -> Dict:
        """Validate the `original_parameters` configuration dictionary.

        Args:
            original_parameters: Dictionary containing the instructions to be validated.
            historical: Option to define if the `original_parameters` should have its
              mandatory keys evaluated for historical or incremental ingestion.

        Raises:
            ValueError: Raised if:
              - the `original_parameters` length equals 0.
              - the type of the value of each `original_parameters` key does not match
                the predefined accepted types:
                - for historical ingestion:
                    ```yaml
                    "col_to_filter_by": str,
                    "start_dt": str,
                    "end_dt": str,
                    ```
                - for incremental ingestion:
                    ```yaml
                    "col_to_filter_by": str,
                    "existing_data_df": str,
                    ```
              - the `col_to_filter_by` parameter is not present in the
                `original_parameters`.

        Returns:
            parsed_parameters: Parsed parameters dict.
        """
        self.logger.info(
            "`original_parameters` before parsing:\n%s", pformat(original_parameters)
        )

        if historical:
            ACCEPTED_KEYS_AND_TYPES = {
                "col_to_filter_by": str,
                "start_dt": str,
                "end_dt": str,
            }
        else:
            ACCEPTED_KEYS_AND_TYPES = {
                "col_to_filter_by": str,
                "existing_data_df": str,
            }

        # let the user knows which parameters are divergente from the expected list
        self._log_inconsistent_keys(original_parameters, ACCEPTED_KEYS_AND_TYPES)

        # cast and get existing keys from the received original_parameters_dict
        # with parsed keys/values
        parsed_parameters = {}
        for parameter_name in original_parameters:
            # get the parameter value from original_parameters and casts it, according
            # to the ACCEPTED_KEYS_AND_TYPES defined types
            try:
                if parameter_name in ACCEPTED_KEYS_AND_TYPES:
                    # get desired_type from our dict and cast_type (may be: str or bool)
                    desired_type = ACCEPTED_KEYS_AND_TYPES.get(parameter_name)
                    parameter_value = original_parameters.get(parameter_name)
                    # call desired_type as a function: str() or bool()
                    parsed_parameters[parameter_name] = desired_type(parameter_value)

            except ValueError as exc:
                raise ValueError(
                    f"{parameter_name} value in `parameters_dict` does not match the "
                    f"required type `{desired_type}`, got the value: {parameter_value}"
                ) from exc

        self.logger.info("Parameters after parsing:\n%s", pformat(parsed_parameters))

        # check existence of at least one parameter to process
        if len(parsed_parameters.keys()) == 0:
            raise ValueError(
                "Empty parameters after parsing keys/values, nothing to do. "
                "Check previous warning messages. Stopping execution"
            )

        # check if we have minimal parameters to process historical / incremental
        if "col_to_filter_by" not in parsed_parameters.keys():
            raise ValueError(
                "At least these parameters are necessary to allow historical"
                " processing. "
                "operation: [col_to_filter_by]. Stopping execution"
            )

        return parsed_parameters

    def _convert_column_to_date(
        self, input_df: pyspark.sql.DataFrame, desired_date_col: str
    ) -> pyspark.sql.DataFrame:
        """Converts column to date type.

        Args:
            input_df: Dataframe containing data.
            desired_date_col: Column to be converted to date type.

        Returns:
            None
        """
        original_data_type = str(input_df.schema[desired_date_col].dataType)
        if original_data_type not in ("DateType", "TimestampType"):
            input_df = input_df.withColumn(
                desired_date_col, f.to_date(desired_date_col)
            )
            self.logger.info(
                "Column `%s` is not DateType or TimestampType, got `%s`. "
                "Trying conversion to date",
                desired_date_col,
                original_data_type,
            )

        return input_df

    def _log_inconsistent_keys(
        self, original_parameters: dict, accepted_keys_and_types: dict
    ) -> None:
        """Logs to the console inconsistent keys received in parameters.

        Args:
            original_parameters: Parameters passed by user.
            accepted_keys_and_types: Parameters expected by function.

        Returns:
            None
        """
        missing_parameters = sorted(
            set(accepted_keys_and_types) - set(original_parameters)
        )

        if missing_parameters:
            self.logger.info("Unspecified parameters:\n%s", missing_parameters)

        unrecognized_parameters = sorted(
            set(original_parameters) - set(accepted_keys_and_types)
        )

        if unrecognized_parameters:
            self.logger.warning("Unrecognized parameters:\n%s", unrecognized_parameters)
