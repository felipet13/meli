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
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from feature_generation.v1.core.utils.melt import melt


@pytest.fixture()
def wide_df(spark):
    schema = StructType(
        [
            StructField("product", StringType()),
            StructField("uk", IntegerType()),
            StructField("us", IntegerType()),
            StructField("in", IntegerType()),
            StructField("jp", IntegerType()),
        ]
    )

    data = [
        (
            "orange",
            12,
            23,
            17,
            12,
        ),
        ("mango", 5, 2, 25, 1),
    ]

    return spark.createDataFrame(data, schema=schema)


def test_melt(wide_df):
    input_df = wide_df

    unpivot_col_list = ["uk", "us", "in", "jp"]
    output_df = melt(
        input_df,
        key_cols=["product"],
        unpivot_col_list=unpivot_col_list,
        output_key_column="country",
        output_value_column="quantity",
    )

    assert set(output_df.columns) == {
        "product",
        "country",
        "quantity",
    }
    assert input_df.count() * len(unpivot_col_list) == output_df.count()
