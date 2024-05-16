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
import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from feature_generation.v1.core.utils.upsert import upsert


@pytest.fixture()
def old_df(spark):
    schema = StructType(
        [
            StructField("join_key", StringType()),
            StructField("attr_1", IntegerType()),
            StructField("attr_2", IntegerType()),
        ]
    )

    data = [
        ("orange", 1, 3),
        ("mango", 2, 4),
    ]

    return spark.createDataFrame(data, schema=schema)


@pytest.fixture()
def new_df(spark):
    schema = StructType(
        [
            StructField("join_key", StringType()),
            StructField("attr_1", IntegerType()),
            StructField("attr_2", IntegerType()),
        ]
    )

    data = [
        ("orange", 1, 2),
        ("apple", 3, 4),
    ]

    return spark.createDataFrame(data, schema=schema)


@pytest.fixture()
def new_df_schema(spark):
    schema = StructType(
        [
            StructField("join_key", StringType()),
            StructField("attr_1", IntegerType()),
            StructField("attr_3", IntegerType()),
        ]
    )

    data = [
        ("orange", 2, 2),
        ("apple", 3, 4),
    ]
    return spark.createDataFrame(data, schema=schema)


def test_upsert(old_df, new_df):
    new_df = upsert(df_old=old_df, df_new=new_df, join_key="join_key")

    assert new_df.count() == 3
    assert new_df.columns == ["join_key", "attr_1", "attr_2"]
    assert (
        new_df.filter(f.col("join_key") == f.lit("orange"))
        .select("attr_2")
        .collect()[0][0]
        == 2
    )


def test_upsert_uneven_schema(old_df, new_df_schema):
    new_df = upsert(df_old=old_df, df_new=new_df_schema, join_key="join_key")

    assert new_df.count() == 3
    assert new_df.columns == ["join_key", "attr_1", "attr_2", "attr_3"]
    assert (
        new_df.filter(f.col("join_key") == f.lit("orange"))
        .select("attr_1")
        .collect()[0][0]
        == 2
    )
