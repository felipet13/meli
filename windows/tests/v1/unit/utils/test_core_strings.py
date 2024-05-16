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

import pandas as pd
import pyspark.sql.functions as f
import pytest
from pyspark.sql.types import StringType, StructField, StructType

from feature_generation.v1.core.utils.strings import (
    keep_alphanumeric,
    map_values_from_dict,
    mask_string,
    regex_map_values_from_dict,
    remove_accents,
    sequential_regexp_replace,
)


@pytest.mark.parametrize(
    "word",
    [
        ({"word": "résumé"}, {"word": "resume"}),
        ({"word": "tête-à-tête"}, {"word": "tete-a-tete"}),
        ({"word": "acute (é)"}, {"word": "acute (e)"}),
        ({"word": ""}, {"word": ""}),  # empty string
        ({"word": None}, {"word": None}),  # null
        ({"word": "grave (è)"}, {"word": "grave (e)"}),
        ({"word": "circumflex (â, î or ô)"}, {"word": "circumflex (a, i or o)"}),
        ({"word": "tilde (ñ)"}, {"word": "tilde (n)"}),
        (
            {
                "word": "dieresis (ü or ï – the same symbol is used for two different purposes)"
            },
            {
                "word": "dieresis (u or i – the same symbol is used for two different purposes)"
            },
        ),
        ({"word": "cedilla (ç)"}, {"word": "cedilla (c)"}),
        ({"word": "123"}, {"word": "123"}),
    ],
)
def test_remove_accents(word, spark):
    # clean conversion to spark dataframe
    schema = StructType([StructField("word", StringType(), True)])
    df = spark.createDataFrame(pd.DataFrame(word), schema=schema)

    df = df.select(
        remove_accents(f.col("word")).alias("clean_word")  # pylint: disable=no-member
    )

    assert df.dropDuplicates().count() == 1


def test_map_values_from_dict(get_sample_spark_data_frame, get_mapping_dictionary):
    df = get_sample_spark_data_frame
    df = df.withColumn(
        "new_value", map_values_from_dict("occupation", get_mapping_dictionary, False)
    ).withColumn(
        "new_value_coalesce", map_values_from_dict("occupation", get_mapping_dictionary)
    )

    new_values = {x[0] for x in df.select("new_value").dropna().collect()}
    num_nulls = df.filter(f.col("new_value").isNull()).count()
    coalesce_values = {x[0] for x in df.select("new_value_coalesce").dropna().collect()}

    mapped_values = {value for k, value in get_mapping_dictionary.items()}

    assert new_values == mapped_values
    assert num_nulls == 2
    assert coalesce_values == mapped_values.union(
        {"daata translator", "Mother of Dragons"}
    )


@pytest.mark.parametrize(
    "input_data, expected_data, replacement_dict",
    [
        ("abc", "cbc", {"a": "c"}),
        ("abc", "ccc", {"a": "c", "b": "c"}),
        ("abcd", "dddd", {"a": "b", "b": "c", "c": "d"}),
        ("abcd", "dddd", {"b": "c", "a": "b", "c": "d"}),
        ("number", "numb", {"e": "", "r": ""}),
        (None, None, {"pointless": "perhaps"}),
        ("Watch it", None, {"a": None}),
    ],
)
def test_sequential_regexp_replace(input_data, expected_data, replacement_dict, spark):
    schema = StructType(
        [
            StructField("input", StringType(), True),
            StructField("expected", StringType(), True),
        ]
    )

    df = spark.createDataFrame([(input_data, expected_data)], schema=schema)

    result = df.select(
        sequential_regexp_replace("input", replacement_dict).alias("masked"),
        f.col("expected"),
    ).collect()

    assert result[0][0] == result[0][1]


@pytest.mark.parametrize(
    "data",
    [
        ("spark@spark.com", "aaaaa@aaaaa.aaa"),
        ("spark999@spark.com", "aaaaa000@aaaaa.aaa"),
        ("CaPiTaL1234LeTTEr", "AaAaAaA0000AaAAAa"),
        ("a1b2b3b4!", "a0a0a0a0!"),
        ("L0LN00BH4x0r!", "A0AA00AA0a0a!"),
        (None, None),
        ("!!!!", "!!!!"),
    ],
)
def test_mask_string(data, spark):
    schema = StructType(
        [
            StructField("input", StringType(), True),
            StructField("expected", StringType(), True),
        ]
    )

    df = spark.createDataFrame([data], schema=schema)

    result = df.select(
        mask_string("input").alias("masked"), f.col("expected")
    ).collect()

    assert result[0][0] == result[0][1]


@pytest.mark.parametrize(
    "data",
    [
        ("123-abcjw:, .@! eiw", "123abcjw  eiw"),
        ("abc_xyz: @1234", "abc_xyz 1234"),
    ],
)
def test_keep_alphanumric(data, spark):
    schema = StructType(
        [
            StructField("input", StringType(), True),
            StructField("expected", StringType(), True),
        ]
    )
    df = spark.createDataFrame([data], schema=schema)

    result = df.select(
        keep_alphanumeric("input").alias("cleaned"), f.col("expected")
    ).collect()

    assert result[0][0] == result[0][1]


@pytest.mark.parametrize(
    "input_data, expected_data, replacement_dict",
    [
        ("aa", "a", {"column": "input", "mapping": {"a": "aa"}}),
        ("aab", "a", {"column": "input", "mapping": {"a": "a.*b"}}),
        (
            "aab",
            "other",
            {"column": "input", "mapping": {"a": "bla"}, "other": "other"},
        ),
        (
            "aab",
            "aab",
            {"column": "input", "mapping": {"a": "bla"}, "coalesce": True},
        ),
    ],
)
def test_regex_map_values_from_dict(input_data, expected_data, replacement_dict, spark):
    schema = StructType(
        [
            StructField("input", StringType(), True),
            StructField("expected", StringType(), True),
        ]
    )

    df = spark.createDataFrame([(input_data, expected_data)], schema=schema)

    result = df.select(
        regex_map_values_from_dict(**replacement_dict).alias("masked"),
        f.col("expected"),
    ).collect()

    assert result[0][0] == result[0][1]
