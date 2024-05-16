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

import pyspark.sql.functions as f
import pytest
from pyspark import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, StringType, StructField, StructType

from feature_generation.v1.core.features.join_dataframes import (
    join_dataframes_with_spine,
)


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.config("spark.sql.shuffle.partitions", 1).getOrCreate()


@pytest.fixture(scope="module")
def spine_df(spark):
    spine_df = spark.createDataFrame(
        [
            Row(element_id="id_0", observation_dt=datetime(2014, 1, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 2, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 3, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 4, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 5, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 6, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 7, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 8, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 9, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 10, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 11, 1)),
            Row(element_id="id_0", observation_dt=datetime(2014, 12, 1)),
            Row(element_id="id_1", observation_dt=datetime(2014, 1, 1)),
            Row(element_id="id_2", observation_dt=datetime(2014, 2, 1)),
            Row(element_id="id_3", observation_dt=datetime(2014, 3, 1)),
            Row(element_id="id_4", observation_dt=datetime(2014, 4, 1)),
            Row(element_id="id_5", observation_dt=datetime(2014, 5, 1)),
            Row(element_id="id_6", observation_dt=datetime(2014, 6, 1)),
            Row(element_id="id_7", observation_dt=datetime(2014, 7, 1)),
            Row(element_id="id_8", observation_dt=datetime(2014, 8, 1)),
            Row(element_id="id_9", observation_dt=datetime(2014, 9, 1)),
            Row(element_id="id_10", observation_dt=datetime(2014, 10, 1)),
            Row(element_id="id_11", observation_dt=datetime(2014, 11, 1)),
            Row(element_id="id_12", observation_dt=datetime(2014, 12, 1)),
        ],
        schema=StructType(
            [
                StructField("element_id", StringType(), True),
                StructField("observation_dt", DateType(), True),
            ]
        ),
    )
    return spine_df


default_join = ["element_id", "observation_dt"]


def test_join_together_with_spine_as_only_df(spine_df):
    join_dataframes_with_spine(spine_df=spine_df, default_join_keys=default_join)


def test_join_together_joins_correctly(spine_df):
    join_df1 = spine_df.withColumn("feature1", f.lit(1))
    join_df2 = spine_df.withColumn("feature2", f.lit(2))
    join_dfs = {"df1": join_df1, "df2": join_df2}

    merged_df = join_dataframes_with_spine(
        spine_df=spine_df, default_join_keys=default_join, **join_dfs
    )
    assert merged_df.count() == spine_df.count()


def test_alternative_join(spine_df):
    join_df3 = spine_df.select("element_id").distinct().withColumn("feature3", f.lit(3))
    join_dfs = {"df1": join_df3}

    merged_df = join_dataframes_with_spine(
        spine_df=spine_df,
        default_join_keys=default_join,
        alternative_join_mapping={"df1": {"join_keys": "element_id"}},
        **join_dfs,
    )

    assert merged_df.count() == spine_df.count()


def test_join_together_with_join_type(spine_df):
    join_df1 = spine_df.withColumn("feature1", f.lit(1))
    join_df2 = spine_df.withColumn("feature2", f.lit(2))
    join_dfs = {"df1": join_df1, "df2": join_df2}
    alternative_join_mapping = {"df1": {"join_type": "inner"}}

    merged_df = join_dataframes_with_spine(
        spine_df=spine_df,
        default_join_keys=default_join,
        alternative_join_mapping=alternative_join_mapping,
        **join_dfs,
    )
    assert merged_df.count() == spine_df.count()
