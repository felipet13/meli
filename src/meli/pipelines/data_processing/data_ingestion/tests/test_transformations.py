# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organization
# and QuantumBlack, and any unauthorized use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organization with the prior written
# permission of QuantumBlack.
# pylint: disable=invalid-name
"""data_integration tests."""

import logging
from datetime import date, datetime

import chispa
import pytest
from data_integration.tests.v0.conftest import generate_spark_dataframe
from data_integration.v0.core.utils.transformations import (
    dataframe_sampling,
    regex_replace_values,
)
from data_integration.v0.nodes import ingestor
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DateType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


class TestDatatransformation:  # pylint: disable=R0904
    """Test class."""

    @staticmethod
    @pytest.fixture
    def exp_rename_df():
        """Test renaming function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("date_of_birth", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", TimestampType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                datetime(2022, 1, 1, 0, 0, 0, 0),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                datetime(2022, 1, 2, 0, 0, 0, 0),
                " Brazilian ",
                20.0,
            ),
            (
                "c003",
                "1991-11-30",
                date(2022, 1, 1),
                None,
                "English",
                30.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_rename_operation(
        input_df_transformation: DataFrame,
        exp_rename_df: DataFrame,
    ):
        """Test rename operation."""

        instructions = {
            "rename_columns": {
                "customer_id": "customer_id",
                "joining_dt": "joining_dt",
                "birth_dt": "date_of_birth",
                "update_dt": "update_dt",
            }
        }
        output_df = ingestor.transform_data(input_df_transformation, instructions)

        chispa.assert_df_equality(
            output_df,
            exp_rename_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_cast_df():
        """Test casting function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", IntegerType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                1640995200,
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                1641081600,
                " Brazilian ",
                20.0,
            ),
            (
                "c003",
                "1991-11-30",
                date(2022, 1, 1),
                None,
                "English",
                30.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_cast_operation(
        input_df_transformation: DataFrame,
        exp_cast_df: DataFrame,
    ):
        """Test cast operation."""

        instructions = {
            "cast_columns": {
                "customer_id": "str",
                "joining_dt": "date",
                "update_dt": "int",
            }
        }
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_cast_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_trim_all_string_df():
        """Test trim string columns function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", TimestampType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                datetime(2022, 1, 1, 0, 0, 0, 0),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                datetime(2022, 1, 2, 0, 0, 0, 0),
                "Brazilian",
                20.0,
            ),
            (
                "c003",
                "1991-11-30",
                date(2022, 1, 1),
                None,
                "English",
                30.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_trim_all_string_operation(
        input_df_transformation: DataFrame,
        exp_trim_all_string_df: DataFrame,
        caplog,
    ):
        """Test trim operation."""
        caplog.set_level(logging.INFO)

        instructions_dict = {"trim_all_string_columns": None}
        output_df = ingestor.transform_data(input_df_transformation, instructions_dict)

        chispa.assert_df_equality(
            output_df,
            exp_trim_all_string_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_trim_df():
        """Test trim function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", TimestampType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                datetime(2022, 1, 1, 0, 0, 0, 0),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                datetime(2022, 1, 2, 0, 0, 0, 0),
                "Brazilian",
                20.0,
            ),
            (
                "c003",
                "1991-11-30",
                date(2022, 1, 1),
                None,
                "English",
                30.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_trim_operation(
        input_df_transformation: DataFrame,
        exp_trim_df: DataFrame,
    ):
        """Test trim operation."""

        instructions_dict = {"trim_columns": ["nationality"]}
        output_df = ingestor.transform_data(input_df_transformation, instructions_dict)

        chispa.assert_df_equality(
            output_df,
            exp_trim_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_fillna_df():
        """Test fillna function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", TimestampType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                datetime(2022, 1, 1, 0, 0, 0, 0),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                datetime(2022, 1, 2, 0, 0, 0, 0),
                " Brazilian ",
                20.0,
            ),
            (
                "c003",
                "1991-11-30",
                date(2022, 1, 1),
                datetime(2020, 1, 1, 0, 0, 0, 0),
                "English",
                30.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_fillna_operation(
        input_df_transformation: DataFrame,
        exp_fillna_df: DataFrame,
    ):
        """Test fillna operation."""

        instructions = {"fill_na": {"update_dt": "2020-01-01 00:00:00"}}
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_fillna_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_drop_null_values_df():
        """Test drop_null_values function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", TimestampType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                datetime(2022, 1, 1, 0, 0, 0, 0),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                datetime(2022, 1, 2, 0, 0, 0, 0),
                " Brazilian ",
                20.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_drop_null_values_operation(
        input_df_transformation: DataFrame,
        exp_drop_null_values_df: DataFrame,
    ):
        """Test drop_null_values operation."""

        instructions = {"drop_null_values": ["update_dt"]}
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_drop_null_values_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_sql_expression_df():
        """Test sql_expression function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", StringType(), True),
                StructField("update_dt", TimestampType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-12-10",
                2020,
                datetime(2022, 1, 1, 0, 0, 0, 0),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-12-10",
                2022,
                datetime(2022, 1, 2, 0, 0, 0, 0),
                " Brazilian ",
                20.0,
            ),
            (
                "c003",
                "1991-12-10",
                2022,
                None,
                "English",
                30.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_sql_expression_operation(
        input_df_transformation: DataFrame,
        exp_sql_expression_df: DataFrame,
    ):
        """Test apply_sql_expression operation."""

        instructions = {
            "apply_sql_expression": {
                "joining_dt": "date_format(joining_dt, 'y')",
                "birth_dt": "date_format(date_add(to_date(birth_dt), 10), "
                "'yyyy-MM-dd')",
            }
        }
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_sql_expression_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_drop_columns_df():
        """Test drop_columns function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", TimestampType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                datetime(2022, 1, 1, 0, 0, 0, 0),
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                datetime(2022, 1, 2, 0, 0, 0, 0),
            ),
            (
                "c003",
                "1991-11-30",
                date(2022, 1, 1),
                None,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_drop_columns_operation(
        input_df_transformation: DataFrame,
        exp_drop_columns_df: DataFrame,
    ):
        """Test drop_columns operation."""

        instructions = {"drop_columns": ["nationality", "taxes_paid"]}
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_drop_columns_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_drop_duplicates_df():
        """Test drop_duplicates function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", TimestampType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                datetime(2022, 1, 1, 0, 0, 0, 0),
                "Italian",
                10.0,
            ),
            (
                "c002",
                "1990-11-30",
                date(2022, 1, 1),
                datetime(2022, 1, 2, 0, 0, 0, 0),
                " Brazilian ",
                20.0,
            ),
            # ("c003", "1991-11-30", date(2022, 1, 1), None, "English", 30.0,),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_drop_duplicates_operation(
        input_df_transformation: DataFrame,
        exp_drop_duplicates_df: DataFrame,
    ):
        """Test drop_duplicates operation."""

        instructions = {
            "drop_duplicates": ["joining_dt"],
        }
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_drop_duplicates_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_keep_columns_df():
        """Test renaming function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("joining_dt", DateType(), True),
            ]
        )

        data = [
            (
                "c001",
                date(2020, 1, 1),
            ),
            (
                "c002",
                date(2022, 1, 1),
            ),
            (
                "c003",
                date(2022, 1, 1),
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_keep_columns_operation(
        input_df_transformation: DataFrame,
        exp_keep_columns_df: DataFrame,
    ):
        """Test keep operation."""

        instructions = {"keep_columns": ["customer_id", "joining_dt"]}
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_keep_columns_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_drop_rows_df():
        """Test renaming function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", TimestampType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                datetime(2022, 1, 1, 0, 0, 0, 0),
                "Italian",
                10.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_drop_rows_operation(
        input_df_transformation: DataFrame,
        exp_drop_rows_df: DataFrame,
    ):
        """Test rename operation."""

        instructions = {
            "drop_rows": {
                "drop_invalid_update_dts": "CASE WHEN update_dt IS NULL THEN True "
                "ELSE NULL END",
                "drop_invalid_customers_ending_with_2": "CASE WHEN customer_id LIKE "
                "'%2' THEN True ELSE NULL END",
            }
        }
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_drop_rows_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_keep_rows_df():
        """Test keep rows function of the package."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("birth_dt", StringType(), True),
                StructField("joining_dt", DateType(), True),
                StructField("update_dt", TimestampType(), True),
                StructField("nationality", StringType(), True),
                StructField("taxes_paid", FloatType(), True),
            ]
        )

        data = [
            (
                "c001",
                "1989-11-30",
                date(2020, 1, 1),
                datetime(2022, 1, 1, 0, 0, 0, 0),
                "Italian",
                10.0,
            ),
            (
                "c003",
                "1991-11-30",
                date(2022, 1, 1),
                None,
                "English",
                30.0,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_keep_rows_operation(
        input_df_transformation: DataFrame,
        exp_keep_rows_df: DataFrame,
    ):
        """Test keep rows operation."""

        instructions = {
            "keep_rows": {
                "keep_invalid_update_dts": "CASE WHEN update_dt IS NULL THEN True "
                "ELSE NULL END",
                "keep_valid_customers_ending_with_1": "CASE WHEN customer_id = 'c001' "
                "THEN True ELSE NULL END",
            }
        }
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_keep_rows_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )

    @staticmethod
    @pytest.fixture
    def exp_all_operations_df():
        """Test all operations to make sure it runs seamlessly."""

        EXP_SCHEMA = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("joining_dt", StringType(), True),
                StructField("update_dt", IntegerType(), True),
            ]
        )

        data = [
            (
                "c001",
                "2020",
                1640995200,
            ),
        ]
        df = generate_spark_dataframe(data, EXP_SCHEMA)

        return df

    @staticmethod
    def test_all_operations(
        input_df_transformation: DataFrame,
        exp_all_operations_df: DataFrame,
    ):
        """Test all operations."""

        instructions = {
            "rename_columns": {
                "customer_id": "customer_id",
                "joining_dt": "joining_dt",
                "birth_dt": "date_of_birth",
                "update_dt": "update_dt",
            },
            "cast_columns": {
                "customer_id": "str",
                "joining_dt": "date",
                "update_dt": "int",
            },
            "trim_columns": ["nationality"],
            "drop_null_values": ["update_dt"],
            "apply_sql_expression": {
                "joining_dt": "date_format(joining_dt, 'y')",
                "date_of_birth": "date_format(date_add(to_date(date_of_birth), 10), "
                "'yyyy-MM-dd')",
            },
            "drop_columns": ["nationality", "taxes_paid"],
            "drop_duplicates": ["joining_dt"],
            "keep_columns": ["customer_id", "joining_dt", "update_dt"],
            "keep_rows": {
                "drop_invalid_customers_ending_with_1": "CASE WHEN customer_id "
                "LIKE '%1' THEN True ELSE NULL END",
            },
            "drop_rows": {
                "drop_update_dt_greater_than_threshold": "CASE "
                "WHEN update_dt > 1641006000 THEN True ELSE NULL END",
            },
        }
        output_df = ingestor.transform_data(
            input_df_transformation,
            instructions,
        )

        chispa.assert_df_equality(
            output_df,
            exp_all_operations_df,
            ignore_row_order=True,
            ignore_column_order=True,
            ignore_nullable=True,
        )


def test_regex_replace_values():
    """Test regex_replace_values function."""
    # Create test DataFrame
    data = [("1", "email#"), ("2", "big_123"), ("3", "bigger$"), ("4", "bigger$")]
    columns = ["id", "label"]
    df = generate_spark_dataframe(data, columns)

    # Define dictionary for column regex replacement
    columns_dict = {
        "label": {r"big": "_PARTIAL_", r"[^a-zA-Z0-9_.-]": "_WILD_", "\\d+": "_NUM_"}
    }

    # Apply regex replacements to DataFrame
    df_transformed = regex_replace_values(df, columns_dict)

    # Check transformed DataFrame
    expected_data = [
        ("1", "email_WILD_"),
        ("2", "_PARTIAL___NUM_"),
        ("3", "_PARTIAL_ger_WILD_"),
        ("4", "_PARTIAL_ger_WILD_"),
    ]
    expected_df = generate_spark_dataframe(expected_data, columns)
    assert sorted(df_transformed.collect()) == sorted(expected_df.collect())


def test_dataframe_sampling():
    """Test dataframe_sampling function."""
    # Define a schema for the test DataFrame
    schema = StructType([StructField("num", IntegerType())])

    # Create a test DataFrame with 100 rows of sequential numbers
    data = [(i,) for i in range(1, 101)]
    df = generate_spark_dataframe(data, schema)

    # Test sampling without replacement
    sample_df = dataframe_sampling(df, withReplacement=False, fraction=0.5, seed=42)
    assert sample_df.count() == 39
    # Test sampling with replacement
    sample_df = dataframe_sampling(df, withReplacement=True, fraction=0.5, seed=42)
    assert sample_df.count() == 55
