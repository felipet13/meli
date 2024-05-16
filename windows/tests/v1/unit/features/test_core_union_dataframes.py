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

from datetime import datetime

import pytest
from pyspark import Row
from pyspark.sql.types import DateType, StringType, StructField, StructType

from feature_generation.v1.core.features.union_dataframes import reduce_union_dataframes


@pytest.fixture()
def df_one(spark):
    spine_df = spark.createDataFrame(
        [
            Row(element_id="id_0", observation_dt=datetime(2014, 1, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 2, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 3, 1)),
        ],
        schema=StructType(
            [
                StructField("element_id", StringType(), True),
                StructField("observation_dt", DateType(), True),
            ]
        ),
    )
    return spine_df


@pytest.fixture()
def df_two(spark):
    spine_df = spark.createDataFrame(
        [
            Row(element_id="id_1", observation_dt=datetime(2014, 1, 1)),
            Row(element_id="id_1", observation_dt=datetime(2014, 2, 1)),
            Row(element_id="id_1", observation_dt=datetime(2014, 3, 1)),
        ],
        schema=StructType(
            [
                StructField("element_id", StringType(), True),
                StructField("observation_dt", DateType(), True),
            ]
        ),
    )
    return spine_df


@pytest.fixture()
def df_three(spark):
    spine_df = spark.createDataFrame(
        [
            Row(observation_dt=datetime(2014, 1, 1), element_id="id_1"),
            Row(observation_dt=datetime(2014, 2, 1), element_id="id_1"),
            Row(observation_dt=datetime(2014, 3, 1), element_id="id_1"),
        ],
        schema=StructType(
            [
                StructField("observation_dt", DateType(), True),
                StructField("element_id", StringType(), True),
            ]
        ),
    )
    return spine_df


@pytest.fixture()
def df_four(spark):
    spine_df = spark.createDataFrame(
        [
            Row(observation_dt=datetime(2014, 1, 1)),
            Row(observation_dt=datetime(2014, 2, 1)),
            Row(observation_dt=datetime(2014, 3, 1)),
        ],
        schema=StructType(
            [
                StructField("observation_dt", DateType(), True),
            ]
        ),
    )
    return spine_df


def test_union_dataframes(df_one, df_two):
    merged_df = reduce_union_dataframes(df_one=df_one, df_two=df_two)
    assert merged_df.count() == 6
    assert merged_df.schema.names == ["element_id", "observation_dt"]


def test_union_by_name_dataframes(df_one, df_three):
    merged_df = reduce_union_dataframes(
        df_one=df_one, df_three=df_three, union_by_name=True
    )
    assert merged_df.count() == 6
    assert merged_df.schema.names == ["element_id", "observation_dt"]


def test_union_by_name_allow_missing_dataframes(df_one, df_four):
    merged_df = reduce_union_dataframes(
        df_one=df_one, df_four=df_four, union_by_name=True, allow_missing_columns=True
    )
    assert merged_df.count() == 6
    assert merged_df.schema.names == ["element_id", "observation_dt"]
