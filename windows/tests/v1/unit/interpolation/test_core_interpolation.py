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

import pyspark.sql.functions as f

from feature_generation.v1.core.interpolation.linear import interpolation_linear


def test_interpolation_linear_integer(mock_range_df):
    df = mock_range_df.withColumn(
        "interpolated", interpolation_linear(start_col="int_start", end_col="int_end")
    )

    results = [x[0] for x in df.select("interpolated").collect()]
    assert results == [1, 2, 3, 4, 5]


def test_interpolation_linear_date(mock_range_df):
    df = mock_range_df.withColumn(
        "interpolated", interpolation_linear(start_col="date_start", end_col="date_end")
    )

    results = [x[0] for x in df.select(f.col("interpolated").cast("string")).collect()]
    assert results == [
        "2020-01-15",
        "2020-01-16",
        "2020-01-17",
        "2020-01-18",
        "2020-01-19",
        "2020-01-20",
    ]


def test_interpolation_linear_date_interval(mock_range_df):
    df = mock_range_df.withColumn(
        "interpolated",
        interpolation_linear(
            start_col="date_start", end_col="date_end", step_size="INTERVAL 2 DAYS"
        ),
    )

    results = [x[0] for x in df.select(f.col("interpolated").cast("string")).collect()]
    assert results == ["2020-01-15", "2020-01-17", "2020-01-19"]
