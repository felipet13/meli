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
"""Test functions."""
# pylint: skip-file
# flake8: noqa
import pytest
from pyspark.sql import functions as f
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from feature_generation.v1.core.tags.expand_tags import (  # noqa: E501
    expand_tags,
    expand_tags_all,
    expand_tags_with_spine,
)


@pytest.fixture()
def tags_df(spark):
    """Sample df."""
    schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField(
                "tags_all",
                ArrayType(
                    StructType(
                        [
                            StructField("tag", StringType(), True),
                            StructField("value", DoubleType(), True),
                        ]
                    )
                ),
            ),
        ]
    )
    data = [
        (
            1,
            [
                ("contain_morning_flag", 1.0),
                ("first_shift_typ_is_morning", 1.0),
                ("shift_count_val", 3.0),
            ],
        ),
        (
            2,
            [
                ("contain_night_flag", 1.0),
                ("first_shift_typ_is_night", 1.0),
                ("shift_count_val", 3.0),
            ],
        ),
        (3, [("first_shift_typ_val", None)]),
        (
            4,
            [
                ("dx_type2_diabetes_typ_is_II", 1.0),
                ("tag_valuable_val", 3.0),
                ("flagergasted_flag", 1.0),
            ],
        ),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture()
def tags_duplicate_df(spark):
    """Sample df."""
    schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField(
                "tags_all",
                ArrayType(
                    StructType(
                        [
                            StructField("tag", StringType(), True),
                            StructField("value", DoubleType(), True),
                        ]
                    )
                ),
            ),
        ]
    )
    data = [
        (
            1,
            [
                ("shift_count_val", 2.0),
            ],
        ),
        (
            1,
            [
                ("tag_valuable_val", 1.0),
                ("shift_count_val", 3.0),
            ],
        ),
        (
            2,
            [
                ("first_shift_typ_val", None),
                ("tag_valuable_val", 10.0),
            ],
        ),
        (
            2,
            [
                ("tag_valuable_val", 3.0),
            ],
        ),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture()
def spine_df(spark):
    """Sample df."""
    schema = StructType(
        [
            StructField("id", IntegerType(), True),
        ]
    )
    data = [(1,), (2,), (3,), (4,), (5,)]
    return spark.createDataFrame(data, schema)


@pytest.fixture()
def list_of_tags():
    list_of_tags = [
        "contain_morning_flag",
        "contain_night_flag",
        "contain_evening_flag",
        "first_shift_typ_is_morning",
        "first_shift_typ_is_night",
        "shift_count_val",
        "dx_type2_diabetes_typ_is_II",
        "tag_valuable_val",
        "flagergasted_flag",
    ]
    return list_of_tags


@pytest.fixture()
def list_of_tags_all():
    list_of_tags = [
        "contain_morning_flag",
        "contain_night_flag",
        "first_shift_typ_val",
        "first_shift_typ_is_morning",
        "first_shift_typ_is_night",
        "shift_count_val",
        "dx_type2_diabetes_typ_is_II",
        "tag_valuable_val",
        "flagergasted_flag",
    ]
    return list_of_tags


def test_expand_tags(tags_df, list_of_tags):
    results = expand_tags(
        df_with_tags=tags_df,
        tags_to_convert=list_of_tags,
        tag_col_name="tags_all",
    )
    assert [
        [v for k, v in x.asDict().items()]
        for x in results.select(*list_of_tags).collect()
    ] == [
        [1.0, None, None, 1.0, None, 3.0, None, None, None],
        [None, 1.0, None, None, 1.0, 3.0, None, None, None],
        [None, None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, 1.0, 3.0, 1.0],
    ]


def test_expand_tags_with_fillna(tags_df, list_of_tags):
    results = expand_tags(
        df_with_tags=tags_df,
        tags_to_convert=list_of_tags,
        tag_col_name="tags_all",
        fillna=0,
    )
    assert [
        [v for k, v in x.asDict().items()]
        for x in results.select(*list_of_tags).collect()
    ] == [
        [1.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0],
    ]


def test_expand_tags_with_agg(tags_duplicate_df):
    results = expand_tags(
        df_with_tags=tags_duplicate_df,
        tags_to_convert=["shift_count_val", "tag_valuable_val", "first_shift_typ_val"],
        tag_col_name="tags_all",
        key_cols=["id"],
        column_instructions={
            "tag_valuable_val_max": f.max("tag_valuable_val"),
            "shift_count_val_sum": f.sum("shift_count_val"),
            "first_shift_typ_val": f.max("first_shift_typ_val"),
        },
    )

    assert [
        [v for k, v in x.asDict().items()]
        for x in results.select(
            *["shift_count_val_sum", "tag_valuable_val_max", "first_shift_typ_val"]
        ).collect()
    ] == [
        [5.0, 1.0, None],
        [None, 10.0, None],
    ]


def test_expand_tags_agg_with_fillna(tags_duplicate_df):
    results = expand_tags(
        df_with_tags=tags_duplicate_df,
        tags_to_convert=["shift_count_val", "tag_valuable_val", "first_shift_typ_val"],
        tag_col_name="tags_all",
        key_cols=["id"],
        column_instructions={
            "tag_valuable_val_max": f.max("tag_valuable_val"),
            "shift_count_val_sum": f.sum("shift_count_val"),
            "first_shift_typ_val": f.max("first_shift_typ_val"),
        },
        fillna=0,
    )

    assert [
        [v for k, v in x.asDict().items()]
        for x in results.select(
            *["shift_count_val_sum", "tag_valuable_val_max", "first_shift_typ_val"]
        ).collect()
    ] == [
        [5.0, 1.0, 0.0],
        [0.0, 10.0, 0.0],
    ]


def test_expand_tags_with_spine(spine_df, tags_df, list_of_tags):
    result = expand_tags_with_spine(
        df_with_tags=tags_df,
        tags_to_convert=list_of_tags,
        tag_col_name="tags_all",
        df_spine=spine_df,
        spine_cols=["id"],
    )
    assert result.count() == spine_df.count()


def test_expand_tags_all(tags_df, list_of_tags_all):
    results = expand_tags_all(
        df_with_tags=tags_df,
        tag_col_name="tags_all",
    )

    assert [
        [v for k, v in x.asDict().items()]
        for x in results.select(*list_of_tags_all).collect()
    ] == [
        [1.0, None, None, 1.0, None, 3.0, None, None, None],
        [None, 1.0, None, None, 1.0, 3.0, None, None, None],
        [None, None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, 1.0, 3.0, 1.0],
    ]
