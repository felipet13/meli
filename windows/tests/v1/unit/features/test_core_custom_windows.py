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

# pylint: skip-file
# flake8: noqa


import pytest
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window

from feature_generation.v1.core.features.custom_windows import (
    complete_max,
    complete_sum,
)
from feature_generation.v1.core.features.windows import generate_window_grid


@pytest.fixture
def mock_customer_value_df(spark):
    """Mock customer value df."""
    schema = StructType(
        [
            StructField("customer_id", StringType()),
            StructField("observ", IntegerType()),
            StructField("value", DoubleType()),
            StructField("flag", IntegerType()),
        ]
    )

    data = [
        ["cuid_001", 0, 10.0, 0],
        ["cuid_001", 1, 20.0, 1],
        ["cuid_001", 2, 30.0, 0],
        ["cuid_001", 3, 40.0, 1],
        ["cuid_002", 0, 80.0, 1],
        ["cuid_002", 1, 60.0, 1],
        ["cuid_002", 2, 40.0, 0],
        ["cuid_002", 3, 20.0, 0],
    ]

    return spark.createDataFrame(data, schema)


@pytest.mark.parametrize(
    "nrows, range_between, expected_output",
    [
        (
            2,
            [1, 2],
            [50.0, 70.0, None, None, 100.0, 60.0, None, None],
        ),
        (
            3,
            [1, 3],
            [90.0, None, None, None, 120.0, None, None, None],
        ),
        (
            2,
            [-2, -1],
            [None, None, 30.0, 50.0, None, None, 140.0, 100.0],
        ),
        (
            3,
            [-3, -1],
            [None, None, None, 60.0, None, None, None, 180.0],
        ),
    ],
)
def test_complete_sum_base(
    nrows, range_between, expected_output, mock_customer_value_df
):
    input_df = mock_customer_value_df

    w = Window.partitionBy("customer_id").orderBy("observ").rangeBetween(*range_between)
    output_df = [
        x[0]
        for x in input_df.withColumn(
            "output", complete_sum(nrows)(input_df["value"]).over(w)
        )
        .select("output")
        .collect()
    ]
    assert output_df == expected_output


@pytest.mark.parametrize(
    "nrows, range_between, expected_output",
    [
        (
            2,
            [1, 2],
            [50.0, 70.0, None, None, 100.0, 60.0, None, None],
        ),
        (
            3,
            [1, 3],
            [90.0, None, None, None, 120.0, None, None, None],
        ),
        (
            2,
            [-2, -1],
            [None, None, 30.0, 50.0, None, None, 140.0, 100.0],
        ),
        (
            3,
            [-3, -1],
            [None, None, None, 60.0, None, None, None, 180.0],
        ),
    ],
)
def test_complete_sum_from_window_grid(
    nrows, range_between, expected_output, mock_customer_value_df
):
    input_df = mock_customer_value_df

    columns = generate_window_grid(
        inputs=["value"],
        funcs=[complete_sum(nrows)],
        windows=[{"partition_by": ["customer_id"], "order_by": ["observ"]}],
        ranges_between=[range_between],
    )
    output_grid_df = [x[0] for x in input_df.select(*columns).collect()]
    assert output_grid_df == expected_output


@pytest.mark.parametrize(
    "nrows, range_between, expected_output",
    [
        (
            2,
            [1, 2],
            [1.0, 1.0, None, None, 1.0, 0.0, None, None],
        ),
        (
            3,
            [1, 3],
            [1.0, None, None, None, 1.0, None, None, None],
        ),
        (
            2,
            [-2, -1],
            [None, None, 1.0, 1.0, None, None, 1.0, 1.0],
        ),
        (
            3,
            [-3, -1],
            [None, None, None, 1.0, None, None, None, 1.0],
        ),
    ],
)
def test_complete_max_base(
    nrows, range_between, expected_output, mock_customer_value_df
):
    input_df = mock_customer_value_df

    w = Window.partitionBy("customer_id").orderBy("observ").rangeBetween(*range_between)
    output_df = [
        x[0]
        for x in input_df.withColumn(
            "output", complete_max(nrows)(input_df["flag"]).over(w)
        )
        .select("output")
        .collect()
    ]
    assert output_df == expected_output


@pytest.mark.parametrize(
    "nrows, range_between, expected_output",
    [
        (
            2,
            [1, 2],
            [1.0, 1.0, None, None, 1.0, 0.0, None, None],
        ),
        (
            3,
            [1, 3],
            [1.0, None, None, None, 1.0, None, None, None],
        ),
        (
            2,
            [-2, -1],
            [None, None, 1.0, 1.0, None, None, 1.0, 1.0],
        ),
        (
            3,
            [-3, -1],
            [None, None, None, 1.0, None, None, None, 1.0],
        ),
    ],
)
def test_complete_max_from_window_grid(
    nrows, range_between, expected_output, mock_customer_value_df
):
    input_df = mock_customer_value_df

    columns = generate_window_grid(
        inputs=["flag"],
        funcs=[complete_max(nrows)],
        windows=[{"partition_by": ["customer_id"], "order_by": ["observ"]}],
        ranges_between=[range_between],
    )
    output_grid_df = [x[0] for x in input_df.select(*columns).collect()]
    assert output_grid_df == expected_output
