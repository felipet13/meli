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
import operator as op

import pyspark.sql.functions as f

from feature_generation.v1.core.tags.generate_tags import create_tags_from_config_select
from feature_generation.v1.core.tags.tags import (
    array_contains_all,
    array_not_contains_all,
    array_rlike,
    arrays_not_overlap,
    arrays_overlap,
    contains_rlike_tags,
    contains_tags,
    convert_tag_to_column,
    count_and_compare_tags,
    dynamic_tag,
    expr_tag,
    extract_all_tag_names,
    extract_tag_value,
    filter_rlike_tags,
    filter_tags,
    isin,
    nth_occurrence,
    one_hot_tag,
    rlike,
    rlike_multi_col,
    tag_array_contains_all,
    tag_array_not_contains_all,
    tag_array_rlike,
    tag_arrays_not_overlap,
    tag_arrays_overlap,
)


def test_extract_all_tag_names(df_sample_with_tags):
    result1 = df_sample_with_tags.select(
        extract_all_tag_names("tags", alias="result1")
    ).collect()
    result2 = df_sample_with_tags.select(
        extract_all_tag_names(f.col("tags"), alias="result2")
    ).collect()

    assert (
        [x[0] for x in result1]
        == [x[0] for x in result2]
        == [
            ["occupation-data"],
            ["occupation-data"],
            ["occupation-data"],
            ["occupation-not-data"],
            ["extra-tag"],
            ["occupation-not-data"],
            ["occupation-data", "extra-tag"],
            ["occupation-not-data"],
        ]
    )


def test_filter_tags(df_sample_with_tags):
    result = df_sample_with_tags.select(
        filter_rlike_tags(
            patterns=["occu", "ext"], tag_col_name="tags", alias="filtered"
        )
    )
    assert [[y.asDict() for y in x[0]] for x in result.collect()] == [
        [{"tag": "occupation-data", "value": 1.0}],
        [{"tag": "occupation-data", "value": 1.0}],
        [{"tag": "occupation-data", "value": 1.0}],
        [{"tag": "occupation-not-data", "value": 1.0}],
        [{"tag": "extra-tag", "value": 1.0}],
        [{"tag": "occupation-not-data", "value": 1.0}],
        [
            {"tag": "occupation-data", "value": 1.0},
            {"tag": "extra-tag", "value": 1.0},
        ],
        [{"tag": "occupation-not-data", "value": 1.0}],
    ]


def test_filter_rlike_tags(df_sample_with_tags):
    result = df_sample_with_tags.select(
        filter_tags(tags="occupation-data", alias="filtered")
    )

    assert [[y.asDict() for y in x[0]] for x in result.collect()] == [
        [{"tag": "occupation-data", "value": 1.0}],
        [{"tag": "occupation-data", "value": 1.0}],
        [{"tag": "occupation-data", "value": 1.0}],
        [],
        [],
        [],
        [{"tag": "occupation-data", "value": 1.0}],
        [],
    ]


def test_contains_tags(df_sample_with_tags):
    result = df_sample_with_tags.filter(
        contains_tags(tags="occupation-data").alias("filtered")
    )
    assert result.count() == 4


def test_contains_rlike_tags(df_sample_with_tags):
    result = df_sample_with_tags.filter(
        contains_rlike_tags(patterns="occupa").alias("filtered")
    )
    assert result.count() == 7


def test_convert_tag_to_column_with_alias(df_sample_with_tags):
    result = df_sample_with_tags.select(
        convert_tag_to_column(
            tag="occupation-data", tag_col_name="tags", alias="occupation-data"
        )
    )

    results = [
        x.asDict().get("occupation-data")
        for x in result.select("occupation-data").collect()
    ]

    results = [x.asDict() if x is not None else None for x in results]

    assert results == [
        {"tag": "occupation-data", "value": 1.0},
        {"tag": "occupation-data", "value": 1.0},
        {"tag": "occupation-data", "value": 1.0},
        None,
        None,
        None,
        {"tag": "occupation-data", "value": 1.0},
        None,
    ]


def test_convert_tag_to_column_without_alias(df_sample_with_tags):
    result = df_sample_with_tags.select(
        convert_tag_to_column(tag="occupation-data", tag_col_name="tags")
    )

    assert result.columns == [
        "filter(tags, lambdafunction((namedlambdavariable().tag IN (occupation-data)), namedlambdavariable()))[0]"
    ]


def test_cast_tag_type(df_sample_with_extracted_tags):
    results = df_sample_with_extracted_tags.select(
        extract_tag_value(tag="tag-xyz", alias="tag-xyz"),
        extract_tag_value(tag="tag-abc", alias="tag-abc"),
    )

    assert results.dtypes == [("tag-xyz", "double"), ("tag-abc", "double")]

    assert [[y for y in x] for x in results.collect()] == [
        [1.0, 2.1],
        [None, 3.7],
        [0.0, None],
        [-1.0, 101.2],
        [None, None],
    ]


def test_isin(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        isin(tag="test_successful", input="name", values=["Tyrion", "Jon", "Arya"]),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 3
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]


def test_rlike(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        rlike(
            tag="test_successful", input="occupation", values=["data", "Data", "daata"]
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 4
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]


def test_rlike_multi_col(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        rlike_multi_col(
            tag="partners_in_data_field",
            inputs=[
                {"input": "name", "values": ["Sansa", "Tyrion"]},
                {
                    "input": "occupation",
                    "values": ["Data", "daata"],
                },
            ],
            operator=op.and_,
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="partners_in_data_field", tag_col_name="test_col")
        ).count()
        == 2
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "_null", "value": 1}],
        [{"tag": "_null", "value": 1}],
        [{"tag": "partners_in_data_field", "value": 1}],
        [{"tag": "_null", "value": 1}],
        [{"tag": "_null", "value": 1}],
        [{"tag": "_null", "value": 1}],
        [{"tag": "partners_in_data_field", "value": 1}],
        [{"tag": "_null", "value": 1}],
    ]


def test_dynamic_tag(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        dynamic_tag(tag="test_successful", expr="age"),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 8
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful", "value": 31.0}],
        [{"tag": "test_successful", "value": 12.0}],
        [{"tag": "test_successful", "value": 65.0}],
        [{"tag": "test_successful", "value": 29.0}],
        [{"tag": "test_successful", "value": 31.0}],
        [{"tag": "test_successful", "value": 27.0}],
        [{"tag": "test_successful", "value": 26.0}],
        [{"tag": "test_successful", "value": 36.0}],
    ]


def test_one_hot_tag(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        one_hot_tag(tag="test_successful", expr="age + 1"),
    )

    assert (
        results.filter(
            contains_rlike_tags(patterns="test_successful", tag_col_name="test_col")
        ).count()
        == 8
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful_32", "value": 1}],
        [{"tag": "test_successful_13", "value": 1}],
        [{"tag": "test_successful_66", "value": 1}],
        [{"tag": "test_successful_30", "value": 1}],
        [{"tag": "test_successful_32", "value": 1}],
        [{"tag": "test_successful_28", "value": 1}],
        [{"tag": "test_successful_27", "value": 1}],
        [{"tag": "test_successful_37", "value": 1}],
    ]


def test_expr_tag(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        expr_tag(tag="test_successful", expr="age > 30"),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 4
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
    ]


def test_tag_array_rlike(df_sample_with_tags):
    results = df_sample_with_tags.withColumn(
        "test_col",
        tag_array_rlike(
            tag="test_successful", input="tags", values=["occupation-data"]
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 4
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]


def test_array_rlike(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        array_rlike(
            tag="test_successful",
            input="house",
            values=["Targaryen", "Stark", "Baratheon"],
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 6
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
    ]


def test_tag_arrays_overlap(df_sample_with_tags):
    results = df_sample_with_tags.withColumn(
        "test_col",
        tag_arrays_overlap(
            tag="test_successful",
            input="tags",
            values=["extra-tag", "occupation-data"],
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 5
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]


def test_arrays_overlap(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        arrays_overlap(
            tag="test_successful",
            input="number",
            values=[1, 3],
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 3
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]


def test_tag_arrays_not_overlap(df_sample_with_tags):
    results = df_sample_with_tags.withColumn(
        "test_col",
        tag_arrays_not_overlap(
            tag="test_successful",
            input="tags",
            values=["occupation-data", "extra-tag"],
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 3
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
    ]


def test_arrays_not_overlap(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        arrays_not_overlap(
            tag="test_successful",
            input="number",
            values=[1, 3],
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 5
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
    ]


def test_tag_array_contains_all(df_sample_with_tags):
    results = df_sample_with_tags.withColumn(
        "test_col",
        tag_array_contains_all(
            tag="test_successful",
            input="tags",
            values=["occupation-data", "extra-tag"],
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 1
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]


def test_array_contains_all(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        array_contains_all(
            tag="test_successful",
            input="number",
            values=[1, 2, 3],
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 1
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]


def test_tag_array_not_contains_all(df_sample_with_tags):
    results = df_sample_with_tags.withColumn(
        "test_col",
        tag_array_not_contains_all(
            tag="test_successful",
            input="tags",
            values=["occupation-data", "extra-tag"],
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 7
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
    ]


def test_array_not_contains_all(df_sample_to_create_tags):
    results = df_sample_to_create_tags.withColumn(
        "test_col",
        array_not_contains_all(
            tag="test_successful",
            input="number",
            values=[1, 2, 3],
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 7
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
    ]


def test_count_and_compare_from_code(df_sample_with_tags):
    results = df_sample_with_tags.withColumn(
        "test_col",
        count_and_compare_tags(
            tag="test_successful",
            input="tags",
            # values={"tag": "extra-tag", "value": 1.0},
            values="extra-tag",
            operator=op.ge,
            y=1,
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 2
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]


def test_count_and_compare_from_config(df_sample_with_tags):
    cfg = [
        count_and_compare_tags(
            tag="test_successful",
            input="tags",
            values="extra-tag",
            operator=op.ge,
            y=1,
        )
    ]

    results = create_tags_from_config_select(
        df_sample_with_tags, cfg, tag_col_name="test_col"
    )
    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 2
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [],
        [],
        [],
        [],
        [{"tag": "test_successful", "value": 1.0}],
        [],
        [{"tag": "test_successful", "value": 1.0}],
        [],
    ]


def test_nth_occurrence(df_sample_with_tags):
    df_with_tags_modified = df_sample_with_tags.withColumn("partition", f.lit("a"))
    results = df_with_tags_modified.withColumn(
        "test_col",
        nth_occurrence(
            tag="test_successful",
            input="tags",
            values="extra-tag",
            partition_by="partition",
            order_by="age",
            n=1,
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 1
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]


def test_nth_occurrence_descending(df_sample_with_tags):
    df_with_tags_modified = df_sample_with_tags.withColumn("partition", f.lit("a"))
    results = df_with_tags_modified.withColumn(
        "test_col",
        nth_occurrence(
            tag="test_successful",
            input="tags",
            values="extra-tag",
            partition_by="partition",
            order_by="age",
            n=1,
            descending=True,
        ),
    )

    assert (
        results.filter(
            contains_tags(tags="test_successful", tag_col_name="test_col")
        ).count()
        == 1
    )
    assert [
        [y.asDict() for y in x[0]] for x in results.select("test_col").collect()
    ] == [
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "test_successful", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
        [{"tag": "_null", "value": 1.0}],
    ]
