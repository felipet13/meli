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
"""Test for interaction feature creation."""
# pylint: skip-file
# flake8: noqa

import logging

import pyspark.sql.functions as f
import pytest
from pyspark.sql.types import IntegerType, StructField, StructType

from feature_generation.v1.core.features.interactions import (
    _extract_elements_in_list,
    create_interaction_features,
)


@pytest.fixture()
def df_with_flags(spark):
    """Sample df."""
    schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("channel_x", IntegerType(), True),
            StructField("channel_y", IntegerType(), True),
            StructField("product_1", IntegerType(), True),
            StructField("product_2", IntegerType(), True),
            StructField("key_message_a", IntegerType(), True),
            StructField("key_message_b", IntegerType(), True),
            StructField("key_message_c", IntegerType(), True),
        ]
    )

    data = [
        (1, 0, 0, 0, 0, 0, 0, 0),
        (2, 0, 1, 0, 1, 0, 0, 1),
        (3, 1, 0, 1, 0, 0, 1, 0),
        (4, 1, 1, 1, 1, 1, 0, 0),
    ]

    return spark.createDataFrame(data, schema)


def test_create_interaction_features(df_with_flags):
    df_with_interaction_features = create_interaction_features(
        df=df_with_flags,
        params_interaction=[
            {
                "channel": ["channel_x", "channel_y"],
                "product": ["product_.*"],
                "key_message": ["key_message_.*"],
            }
        ],
        params_spine_cols=["id"],
    )

    assert (
        df_with_interaction_features.filter(f.col("id") == 4)
        .select("channel_x_product_1_key_message_a")
        .collect()[0][0]
        == 1
    )

    assert df_with_interaction_features.columns == [
        "id",
        "channel_x_product_1_key_message_a",
        "channel_x_product_1_key_message_b",
        "channel_x_product_1_key_message_c",
        "channel_x_product_2_key_message_a",
        "channel_x_product_2_key_message_b",
        "channel_x_product_2_key_message_c",
        "channel_y_product_1_key_message_a",
        "channel_y_product_1_key_message_b",
        "channel_y_product_1_key_message_c",
        "channel_y_product_2_key_message_a",
        "channel_y_product_2_key_message_b",
        "channel_y_product_2_key_message_c",
    ]


def test_extract_elements_in_list(caplog):
    caplog.clear()

    _extract_elements_in_list(
        full_list_of_columns=["channel_x", "channel_y"],
        list_of_regexes=["channel_is.*"],
    )

    assert caplog.record_tuples == [
        (
            "feature_generation.v1.core.features.interactions",
            logging.WARNING,
            "The following regex did not return a result: channel_is.*",
        ),
    ]
