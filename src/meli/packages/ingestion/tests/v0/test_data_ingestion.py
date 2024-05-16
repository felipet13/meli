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
"""data_integration tests."""

import logging
from datetime import date

import chispa
import pytest
from pyspark.sql import DataFrame
from pyspark.sql.types import DateType, FloatType, StringType, StructField, StructType

from data_integration.tests.v0.conftest import generate_spark_dataframe
from data_integration.v0.nodes import ingestor


class TestDataIngestion:
    """Test class."""

    @staticmethod
    @pytest.fixture
    def casted_date_col_df():
        """Expected output df of casting partition column to date."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_of_customer", StringType(), True),
                StructField("update_dt", DateType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                "2022-12-25",
                date(2022, 12, 25),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                "2022-12-31",
                date(2022, 12, 31),
                "Brazilian",
                20.0,
            ),
            (
                "c003",
                "1991-11-30",
                date(2022, 1, 1),
                "2022-10-10",
                date(2022, 10, 10),
                "English",
                30.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_cast_date_col(
        input_df_ingestion: DataFrame, casted_date_col_df: DataFrame, caplog
    ):
        """Test cast to Date operation."""
        caplog.set_level(logging.INFO)

        raw_parameters_dict = {
            "col_to_filter_by": "update_dt",
            "date_col_by_expr": {"update_dt": "update_of_customer"},
        }

        output_df = ingestor.ingest_historical_data(
            raw_df=input_df_ingestion,
            input_parameters_dict=raw_parameters_dict,
        )

        chispa.assert_df_equality(
            output_df,
            casted_date_col_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def expected_incremental_df():
        """Expected output df of casting partition column to date."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_of_customer", StringType(), True),
                StructField("update_dt", DateType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                "2022-12-25",
                date(2022, 12, 25),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                "2022-12-31",
                date(2022, 12, 31),
                "Brazilian",
                20.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_incremental_processing_w_date_casting(
        input_df_ingestion: DataFrame,
        expected_incremental_df: DataFrame,
        existing_df: DataFrame,
        caplog,
    ):
        """Test incremental processing after date casting."""
        caplog.set_level(logging.INFO)

        raw_parameters_dict = {
            "col_to_filter_by": "update_dt",
            "date_col_by_expr": {"update_dt": "update_of_customer"},
        }

        output_df = ingestor.ingest_incremental_data(
            raw_df=input_df_ingestion,
            input_parameters_dict=raw_parameters_dict,
            existing_data_df=existing_df,
        )

        chispa.assert_df_equality(
            output_df,
            expected_incremental_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def expected_filtered_dates_df_historical():
        """Expected output df when filtering df through historical processing."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_of_customer", StringType(), True),
                StructField("update_dt", DateType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                "2022-12-25",
                date(2022, 12, 25),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                "2022-12-31",
                date(2022, 12, 31),
                "Brazilian",
                20.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    # validates parameters in historical processing are filtering correctly
    @staticmethod
    def test_historical_processing_filtering_with_dates(
        input_df_ingestion: DataFrame,
        expected_filtered_dates_df_historical: DataFrame,
        caplog,
    ):
        """Test incremental processing after date casting."""
        caplog.set_level(logging.INFO)

        raw_parameters_dict = {
            "date_col_by_expr": {"update_dt": "update_of_customer"},
            "col_to_filter_by": "update_dt",
            "start_dt": "2022-12-01",
            "end_dt": "2022-12-31",
        }

        output_df = ingestor.ingest_historical_data(
            raw_df=input_df_ingestion,
            input_parameters_dict=raw_parameters_dict,
        )

        chispa.assert_df_equality(
            output_df,
            expected_filtered_dates_df_historical,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )
