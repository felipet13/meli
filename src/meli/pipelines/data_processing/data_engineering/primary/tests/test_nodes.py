from datetime import date

import pytest
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from ..nodes_primary import create_windows


# Define a fixture for SparkSession
@pytest.fixture(scope="module")
def spark_fixture():
    spark = SparkSession.builder.master("local[*]").appName("pytest").getOrCreate()
    yield spark
    spark.stop()


def test_create_windows(spark_fixture):
    # Define schema
    schema = StructType(
        [
            StructField("user_id", IntegerType(), True),
            StructField("value_prop", StringType(), True),
            StructField("date", DateType(), True),
            StructField("customer_tap", IntegerType(), True),
            StructField("paid_by_customer", DoubleType(), True),
        ]
    )

    # Create a simple DataFrame for testing
    data = [
        (1, "A", date(2022, 1, 1), 1, 20.0),
        (1, "A", date(2022, 1, 2), 2, 40.0),
        (1, "A", date(2022, 1, 3), 3, 60.0),
        (1, "A", date(2022, 1, 4), 4, 80.0),
        (1, "A", date(2022, 1, 5), 5, 100.0),
        (2, "B", date(2022, 1, 1), 6, 120.0),
        (2, "B", date(2022, 1, 2), 7, 140.0),
        (2, "B", date(2022, 1, 3), 8, 160.0),
        (2, "B", date(2022, 1, 4), 9, 180.0),
        (2, "B", date(2022, 1, 5), 10, 200.0),
        (3, "C", date(2022, 1, 1), 11, 220.0),
        (3, "C", date(2022, 1, 2), 12, 240.0),
        (3, "C", date(2022, 1, 3), 13, 260.0),
        (3, "C", date(2022, 1, 4), 14, 280.0),
        (3, "C", date(2022, 1, 5), 15, 300.0),
    ]
    df = spark_fixture.createDataFrame(data, schema)

    # Define parameters
    parameters = {"date_col": "date", "last_n_days": 7}

    # Call the function
    result_df = create_windows(df, parameters)[0]

    # Check if the result is a DataFrame
    assert isinstance(result_df, SparkDataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = [
        "user_id",
        "value_prop",
        "date",
        "customer_tap",
        "paid_by_customer",
        "count_customer_saw_print_last_3_weeks",
        "count_customer_tap_print_last_3_weeks",
        "sum_customer_paid_last_3_weeks",
        "count_customer_paid_last_3_weeks",
    ]
    assert result_df.columns == expected_columns

    # Collect the DataFrame into memory
    result_data = result_df.collect()

    # Check the values in the first row
    assert result_data[0]["user_id"] == 1
    assert result_data[0]["value_prop"] == "A"
    assert result_data[0]["date"] == date(2022, 1, 1)
    assert result_data[0]["customer_tap"] == 1
    assert result_data[0]["paid_by_customer"] == 20.0
    assert result_data[0]["count_customer_saw_print_last_3_weeks"] == 0
    assert result_data[0]["count_customer_tap_print_last_3_weeks"] is None
    assert result_data[0]["sum_customer_paid_last_3_weeks"] is None
    assert result_data[0]["count_customer_paid_last_3_weeks"] is None

    # Check the values in the middle row
    assert result_data[7]["user_id"] == 2
    assert result_data[7]["value_prop"] == "B"
    assert result_data[7]["date"] == date(2022, 1, 3)
    assert result_data[7]["customer_tap"] == 8
    assert result_data[7]["paid_by_customer"] == 160.0
    # Add assertions for the windowed columns as needed

    # Check the values in the last row
    assert result_data[-1]["user_id"] == 3
    assert result_data[-1]["value_prop"] == "C"
    assert result_data[-1]["date"] == date(2022, 1, 5)
    assert result_data[-1]["customer_tap"] == 15
    assert result_data[-1]["paid_by_customer"] == 300.0
    # Add assertions for the windowed columns as needed
