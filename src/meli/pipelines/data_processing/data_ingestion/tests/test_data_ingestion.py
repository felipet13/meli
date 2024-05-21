"""data_integration tests."""

import logging
from datetime import date

import chispa
import pytest
from pyspark.sql import DataFrame
from pyspark.sql.types import DateType, FloatType, StringType, StructField, StructType

from ..nodes_data_ingestion import ingest_historical_data, ingest_incremental_data
from .conftest import generate_spark_dataframe


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
    @pytest.fixture
    def expected_incremental_df():
        """Expected output df of casting partition column to date."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
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
                date(2022, 12, 25),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
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
        }

        output_df = ingest_incremental_data(
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
                date(2022, 12, 25),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
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
            "col_to_filter_by": "update_dt",
            "start_dt": "2022-12-01",
            "end_dt": "2022-12-31",
        }

        output_df = ingest_historical_data(
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
