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

from feature_generation.v1.core.utils.bucket import (
    create_buckets,
    create_even_ranged_buckets,
    create_splits,
)


@pytest.fixture
def create_buckets_config():
    bucket_params = {
        "bucket1": [1, 15],
        "bucket2": [15, 25],
        "bucket3": [25, 35],
        "bucket4": [35, 45],
        "bucket5": [45, 60],
    }
    return bucket_params


@pytest.fixture
def create_even_buckets_config():
    bucket_params = {"min_val": 0, "max_val": 80, "num_buckets": 4}
    return bucket_params


def test_create_buckets(get_sample_spark_data_frame, create_buckets_config):
    bucket_params = create_buckets_config
    df = get_sample_spark_data_frame.withColumn(
        "bucket_val", create_buckets("age", bucket_params)
    )

    count = df.groupBy("bucket_val").count().orderBy("bucket_val").collect()
    results = {}
    for row in count:
        results[row["bucket_val"]] = row["count"]
    assert results == {"bucket1": 1, "bucket3": 5, "bucket4": 1, "outliers": 1}


def test_create_even_buckets(get_sample_spark_data_frame, create_even_buckets_config):
    bucket_params = create_even_buckets_config
    df = get_sample_spark_data_frame.withColumn(
        "bucket_val", create_even_ranged_buckets("age", bucket_params)
    )

    count = df.groupBy("bucket_val").count().orderBy("bucket_val").collect()

    results = {}
    for row in count:
        results[row["bucket_val"]] = row["count"]
    assert results == {"0.0_to_20.0": 1, "20.0_to_40.0": 6, "60.0_to_80.0": 1}


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ((1, 14, 4), [1, 4.25, 7.5, 10.75, 14]),
        ((-2.5, 2.5, 5), [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
        ((-10, -5, 5), [-10, -9.0, -8.0, -7.0, -6.0, -5]),
    ],
)
def test_create_splits(test_input, expected):
    assert create_splits(*test_input) == expected
