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

"""Test functions for the array related functions."""

# pylint: skip-file
# flake8: noqa

import pyspark.sql.functions as f
import pytest
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from feature_generation.v1.core.utils.arrays import array_rlike


@pytest.fixture
def mock_df(spark):
    """Mock array df."""
    schema = StructType(
        [
            StructField("date_index", IntegerType(), True),
            StructField("person", StringType(), True),
            StructField("tags", ArrayType(StringType()), True),
        ]
    )

    data = [
        (1, "Jon", []),
        (2, "Jon", ["had_fever", "number"]),
        (3, "Jon", ["visit_gp", "number", "bogus"]),
        (4, "Jon", ["had_tummy_ache", "number"]),
        (5, "Jon", ["had_fever", "number"]),
    ]

    return spark.createDataFrame(data, schema)


def test_array_rlike(mock_df):
    df = mock_df.withColumn("array_rlike", array_rlike(input="tags", pattern="number"))

    assert df.filter(f.size("array_rlike") > 0).count() == 4
