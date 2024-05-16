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

from feature_generation.v1.core.tags.tags import _NULL, contains_tags, filter_tags
from feature_generation.v1.nodes.tags.generate_tags import (
    create_tags_from_config,
    create_tags_from_config_broadcast,
    create_tags_from_config_select,
)


@pytest.fixture
def sample_config_for_broadcast():
    return {
        "config": {
            "occupation": [
                {
                    "object": "feature_generation.v1.core.tags.tags.rlike",
                    "tag": "occupation_data",
                    "input": "occupation",
                    "values": ["Data", "daata"],
                },
                {
                    "object": "feature_generation.v1.core.tags.tags.isin",
                    "tag": "is_em",
                    "input": "occupation",
                    "values": ["Engagement Manager", "EM"],
                },
            ],
            "house": [
                {
                    "object": "feature_generation.v1.core.tags.tags.array_contains_all",
                    "tag": "is_house_stark",
                    "input": "house",
                    "values": "House Stark",
                }
            ],
        }
    }


@pytest.fixture
def sample_config_for_select():
    return {
        "config": [
            {
                "object": "feature_generation.v1.core.tags.tags.rlike",
                "tag": "occupation_data",
                "input": "occupation",
                "values": ["Data", "daata"],
            },
            {
                "object": "feature_generation.v1.core.tags.tags.arrays_overlap",
                "tag": "number-3",
                "input": "number",
                "values": [3],
            },
            {
                "object": "feature_generation.v1.core.tags.tags.isin",
                "tag": "is_em",
                "input": "occupation",
                "values": ["Engagement Manager", "EM"],
            },
            {
                "object": "feature_generation.v1.core.tags.tags.expr_tag",
                "tag": "age-gt-30",
                "expr": "age > 30",
            },
        ]
    }


def test_create_tags_from_config_select(
    df_sample_to_create_tags, sample_config_for_select
):
    df_with_tags = create_tags_from_config_select(
        df=df_sample_to_create_tags,
        config=sample_config_for_select["config"],
    )

    assert df_with_tags.columns == [
        "name",
        "age",
        "occupation",
        "house",
        "number",
        "tags",
    ]
    assert (
        df_with_tags.filter(f.size(filter_tags(tags="age-gt-30")) > f.lit(0)).count()
        == 4
    )
    assert (
        df_with_tags.filter(
            f.size(filter_tags(tags=["occupation_data"])) > f.lit(0)
        ).count()
        == 4
    )
    assert df_with_tags.filter(f.size(filter_tags(tags=_NULL)) > f.lit(0)).count() == 0


def test_create_tags_from_config_sequential(df_sample_to_create_tags):
    df = df_sample_to_create_tags
    config = [
        {
            "object": "feature_generation.v1.core.tags.tags.rlike",
            "tag": "occupation_data",
            "input": "occupation",
            "values": ["Data", "daata"],
        },
        {
            "object": "feature_generation.v1.core.tags.tags.tag_arrays_not_overlap",
            "tag": "occupation_not_data",
            "input": "tags",
            "values": ["occupation_data"],
        },
    ]

    results = create_tags_from_config_select(
        df=df,
        config=config,
        tag_col_name="tags",
        sequential=True,
    )

    assert results.columns == [
        "name",
        "age",
        "occupation",
        "house",
        "number",
        "tags",
    ]

    assert [[y.asDict() for y in x[0]] for x in results.select("tags").collect()] == [
        [{"tag": "occupation_data", "value": 1.0}],
        [{"tag": "occupation_data", "value": 1.0}],
        [{"tag": "occupation_data", "value": 1.0}],
        [{"tag": "occupation_not_data", "value": 1.0}],
        [{"tag": "occupation_not_data", "value": 1.0}],
        [{"tag": "occupation_not_data", "value": 1.0}],
        [{"tag": "occupation_data", "value": 1.0}],
        [{"tag": "occupation_not_data", "value": 1.0}],
    ]


def test_create_tags_from_config_broadcast(
    df_sample_to_create_tags, sample_config_for_broadcast
):
    df_with_tags_config_broadcast = (
        create_tags_from_config_broadcast(
            df=df_sample_to_create_tags, config=sample_config_for_broadcast["config"]
        )
        .withColumn("tags_explode", f.explode(f.col("tags")))
        .drop("tags")
    )

    assert df_with_tags_config_broadcast.columns == [
        "name",
        "age",
        "occupation",
        "house",
        "number",
        "tags_explode",
    ]

    assert [
        x[0].asDict()
        for x in df_with_tags_config_broadcast.select("tags_explode").collect()
    ] == [
        {"tag": "occupation_data", "value": 1},
        {"tag": "occupation_data", "value": 1},
        {"tag": "occupation_data", "value": 1},
        {"tag": "is_house_stark", "value": 1},
        {"tag": "is_em", "value": 1},
        {"tag": "is_house_stark", "value": 1},
        {"tag": "is_house_stark", "value": 1},
        {"tag": "occupation_data", "value": 1},
        {"tag": "is_house_stark", "value": 1},
    ]


def test_create_tags_from_config(
    df_sample_to_create_tags, sample_config_for_broadcast, sample_config_for_select
):
    config = [
        {
            "object": "feature_generation.v1.core.tags.tags.rlike",
            "tag": "occupation_data",
            "input": "occupation",
            "values": ["Data", "daata"],
        },
        {
            "object": "feature_generation.v1.core.tags.tags.isin",
            "tag": "is_em",
            "input": "occupation",
            "values": ["Engagement Manager", "EM"],
        },
        {
            "object": "feature_generation.v1.core.tags.tags.arrays_overlap",
            "tag": "number-3",
            "input": "number",
            "values": [3],
        },
        {
            "object": "feature_generation.v1.core.tags.tags.expr_tag",
            "tag": "age-gt-30",
            "expr": "age > 30",
        },
        {
            "object": "feature_generation.v1.core.tags.tags.expr_tag",
            "tag": "senior_citizen",
            "expr": "age > 60",
        },
        {
            "object": "feature_generation.v1.core.tags.tags.rlike_multi_col",
            "tag": "partners_in_data_field",
            "inputs": [
                {"input": "name", "values": ["Sansa", "Cersei"]},
                {
                    "input": "occupation",
                    "values": ["Data", "daata"],
                },
            ],
        },
    ]

    df_with_tags_from_config = (
        create_tags_from_config(
            df=df_sample_to_create_tags, config=config, verbose=True
        )
        .withColumn("tags_explode", f.explode(f.col("tags")))
        .drop("tags")
    )

    assert df_with_tags_from_config.columns == [
        "name",
        "age",
        "occupation",
        "house",
        "number",
        "_tags_all_broadcast",
        "_tags_all_select",
        "tags_explode",
    ]

    assert [
        x[0].asDict() for x in df_with_tags_from_config.select("tags_explode").collect()
    ] == [
        {"tag": "occupation_data", "value": 1},
        {"tag": "number-3", "value": 1},
        {"tag": "age-gt-30", "value": 1},
        {"tag": "partners_in_data_field", "value": 1},
        {"tag": "occupation_data", "value": 1},
        {"tag": "number-3", "value": 1},
        {"tag": "partners_in_data_field", "value": 1},
        {"tag": "occupation_data", "value": 1},
        {"tag": "number-3", "value": 1},
        {"tag": "age-gt-30", "value": 1},
        {"tag": "senior_citizen", "value": 1},
        {"tag": "partners_in_data_field", "value": 1},
        {"tag": "is_em", "value": 1},
        {"tag": "partners_in_data_field", "value": 1},
        {"tag": "age-gt-30", "value": 1},
        {"tag": "occupation_data", "value": 1},
        {"tag": "partners_in_data_field", "value": 1},
        {"tag": "age-gt-30", "value": 1},
    ]


def test_create_tags_from_config_with_inputs(
    df_sample_to_create_tags, sample_config_for_broadcast, sample_config_for_select
):
    config = [
        {
            "object": "feature_generation.v1.core.tags.tags.rlike",
            "tag": "occupation_data",
            "input": "occupation",
            "values": ["Data", "daata"],
        },
        {
            "object": "feature_generation.v1.core.tags.tags.isin",
            "tag": "is_em",
            "input": "occupation",
            "values": ["Engagement Manager", "EM"],
        },
    ]

    df_with_tags_from_config = (
        create_tags_from_config(
            df=df_sample_to_create_tags, config=config, verbose=True
        )
        .withColumn("tags_explode", f.explode(f.col("tags")))
        .drop("tags")
    )

    assert df_with_tags_from_config.columns == [
        "name",
        "age",
        "occupation",
        "house",
        "number",
        "_tags_all_broadcast",
        "_tags_all_select",
        "tags_explode",
    ]

    assert [
        x[0].asDict() for x in df_with_tags_from_config.select("tags_explode").collect()
    ] == [
        {"tag": "occupation_data", "value": 1},
        {"tag": "occupation_data", "value": 1},
        {"tag": "occupation_data", "value": 1},
        {"tag": "is_em", "value": 1},
        {"tag": "occupation_data", "value": 1},
    ]


def test_create_tags_from_config_without_inputs(
    df_sample_to_create_tags, sample_config_for_broadcast, sample_config_for_select
):
    config = [
        {
            "object": "feature_generation.v1.core.tags.tags.expr_tag",
            "tag": "age-gt-30",
            "expr": "age > 30",
        },
        {
            "object": "feature_generation.v1.core.tags.tags.expr_tag",
            "tag": "senior_citizen",
            "expr": "age > 60",
        },
    ]

    df_with_tags_from_config = (
        create_tags_from_config(
            df=df_sample_to_create_tags, config=config, verbose=True
        )
        .withColumn("tags_explode", f.explode(f.col("tags")))
        .drop("tags")
    )

    assert df_with_tags_from_config.columns == [
        "name",
        "age",
        "occupation",
        "house",
        "number",
        "_tags_all_broadcast",
        "_tags_all_select",
        "tags_explode",
    ]

    assert [
        x[0].asDict() for x in df_with_tags_from_config.select("tags_explode").collect()
    ] == [
        {"tag": "age-gt-30", "value": 1},
        {"tag": "age-gt-30", "value": 1},
        {"tag": "senior_citizen", "value": 1},
        {"tag": "age-gt-30", "value": 1},
        {"tag": "age-gt-30", "value": 1},
    ]
