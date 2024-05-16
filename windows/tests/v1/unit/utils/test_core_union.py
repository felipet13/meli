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

from feature_generation.v1.core.utils.union import lazy_union, reduce_union


def test_reduce_union(get_sample_spark_data_frame):
    list_of_df = [
        get_sample_spark_data_frame,
        get_sample_spark_data_frame,
        get_sample_spark_data_frame,
    ]

    union_of_all = reduce_union(list_of_df)

    assert union_of_all.count() == 8 * 3


def test_reduce_union_by_name(get_sample_spark_data_frame):
    list_of_df = [
        get_sample_spark_data_frame.select("name", "age", "number"),
        get_sample_spark_data_frame.select("age", "number", "name"),
        get_sample_spark_data_frame.select("number", "name", "age"),
    ]

    union_of_all = reduce_union(list_of_df, union_by_name=True)

    assert union_of_all.count() == 8 * 3


def test_lazy_union(get_sample_spark_data_frame):
    list_of_df = [
        get_sample_spark_data_frame.select(
            "name", "age", "number", f.lit(1).alias("c1")
        ),
        get_sample_spark_data_frame.select(
            "age", "number", "name", f.lit("a").alias("c2")
        ),
        get_sample_spark_data_frame.select(
            "number", "name", "age", f.lit(1.0).alias("c3")
        ),
    ]

    union_of_all = lazy_union(list_of_df)

    assert union_of_all.count() == 8 * 3
    assert union_of_all.dtypes == [
        ("name", "string"),
        ("age", "int"),
        ("number", "array<int>"),
        ("c1", "int"),
        ("c2", "string"),
        ("c3", "double"),
    ]
