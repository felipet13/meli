"""Conftest."""

from datetime import date, datetime

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DateType,
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

SCHEMA_INGESTION_ORIG = StructType(
    [
        StructField("customer_id", StringType(), True),
        StructField("birth_dt", StringType(), True),
        StructField("joining_dt", DateType(), True),
        StructField("update_of_customer", StringType(), True),  # Original date column
        StructField("nationality", StringType(), True),
        StructField("taxes_paid", FloatType(), True),
    ]
)

SCHEMA_INGESTION_RENAMED = StructType(
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

SCHEMA_TRANSFORMATION = StructType(
    [
        StructField("customer_id", StringType(), True),
        StructField("birth_dt", StringType(), True),
        StructField("joining_dt", DateType(), True),
        StructField("update_dt", TimestampType(), True),
        StructField("nationality", StringType(), True),
        StructField("taxes_paid", FloatType(), True),
    ]
)


def generate_spark_dataframe(data: list, schema: StructType) -> DataFrame:
    """Generate spark data frame from dataset and schema.

    Args:
        data: List of Pydantic schemas with data to create dataframe.
        schema: A StructType schema definition

    Returns:
        Returns a spark dataframe.
    """
    spark = SparkSession.builder.getOrCreate()

    return spark.createDataFrame(data=data, schema=schema)


@pytest.fixture
def empty_df() -> DataFrame:
    """Create mock customer dataframe for tests."""
    data = []

    df = generate_spark_dataframe(data, SCHEMA_INGESTION_ORIG)

    return df


@pytest.fixture
def input_df_ingestion() -> DataFrame:
    """Create mock customer dataframe for tests."""
    data = [
        (
            "c001",
            "1989-11-30",
            date(2020, 1, 1),
            "2022-12-25",
            "Italian",
            10.0,
        ),
        (
            "c002",
            "1990-11-30",
            date(2022, 1, 1),
            "2022-12-31",
            "Brazilian",
            20.0,
        ),
        (
            "c003",
            "1991-11-30",
            date(2022, 1, 1),
            "2022-10-10",
            "English",
            30.0,
        ),
    ]

    df = generate_spark_dataframe(data, SCHEMA_INGESTION_ORIG)

    return df


@pytest.fixture
def existing_df() -> DataFrame:
    """Create mock customer dataframe for tests."""
    data = [
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

    df = generate_spark_dataframe(data, SCHEMA_INGESTION_RENAMED)

    return df


@pytest.fixture
def input_df_transformation() -> DataFrame:
    """Create mock customer dataframe for tests."""
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

    df = generate_spark_dataframe(data, SCHEMA_TRANSFORMATION)

    return df


@pytest.fixture
def input_df_regex_transformation() -> DataFrame:
    """Create mock customer dataframe for tests."""
    data = [
        (
            "c001",
            "1989-11-30",
            date(2020, 1, 1),
            datetime(2022, 1, 1, 0, 0, 0, 0),
            "It@ali@an",
            10.0,
        ),
        (
            "c002",
            "1990-11-30",
            date(2022, 1, 1),
            datetime(2022, 1, 2, 0, 0, 0, 0),
            " Br@azili@an!",
            20.0,
        ),
        (
            "c003",
            "1991-11-30",
            date(2022, 1, 1),
            None,
            "@En@g#lish",
            30.0,
        ),
    ]

    df = generate_spark_dataframe(data, SCHEMA_TRANSFORMATION)

    return df
