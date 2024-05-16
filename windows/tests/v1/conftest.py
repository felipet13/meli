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

"""Contains fixtures."""

# pylint: skip-file
# flake8: noqa


import datetime
import os
import sys

import pandas as pd
import pytest
from pyspark import Row
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as f
from pyspark.sql.types import (
    ArrayType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from feature_generation.v1.core.features.flags import isin, rlike, rlike_extract

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="module")
def spark():
    """Mock spark session."""
    spark = SparkSession.builder.config("spark.sql.shuffle.partitions", 1).getOrCreate()
    return spark


@pytest.fixture(scope="module")
def get_sample_spark_data_frame(spark):
    """Sample df."""
    schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("occupation", StringType(), True),
            StructField("house", ArrayType(StringType()), True),
            StructField("number", ArrayType(IntegerType()), True),
        ]
    )

    data = [
        ("Gendry", 31, "Data Engineer", ["House Baratheon"], [1, 2, 3]),
        ("Jaime", 12, "Data Scientist", ["House Lannister"], [2, 3, 4]),
        ("Tyrion", 65, "Data Analyst", ["House Lannister", "House Stark"], [3, 4, 5]),
        ("Cersei", 29, "Engagement Manager", ["House Lannister"], [5, 6, 7]),
        ("Jon", 31, "Software Engineer", ["House Targaryen", "House Stark"], [6, 7, 8]),
        ("Arya", 27, "MLE", ["House Stark"], [7, 8, 9]),
        (
            "Sansa",
            26,
            "daata translator",
            ["House Stark", "House Lannister"],
            [8, 9, 10],
        ),
        ("Daenerys", 36, "Mother of Dragons", ["House Targaryen"], [9, 10, 11]),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture(scope="module")
def get_sample_column_instructions():
    """Sample instrctuin for `create_columns_from_config`."""
    sample_column_instructions = [
        rlike("occupation", ["Data", "daata"], "data_related_occupation_based_flag"),
        isin("occupation", ["MLE"], "data_related_hardcoded_flag"),
        rlike_extract(
            "occupation", ["Engineer", "MLE"], "engineer_related_occupation_based_flag"
        ),
        rlike_extract(
            "occupation", ["knight"], "knight_of_the_seven_realms_flag", "_knight_match"
        ),
        rlike(
            "occupation",
            ["Data", "daata"],
            "data_related_occupation_based_age_col",
            "age",
        ),
    ]

    return sample_column_instructions


@pytest.fixture(scope="module")
def get_mapping_dictionary():
    """Mock mapping dictionary."""
    return {
        "Data Engineer": "DE",
        "Data Scientist": "DS",
        "Data Analyst": "BA",
        "Engagement Manager": "EM",
        "MLE": "MLE",
        "Software Engineer": "SE",
    }


@pytest.fixture(scope="module")
def _get_sample_spark_data_frame2(spark):
    """Mock sample df."""
    schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("date", DateType(), True),
            StructField("date_index", IntegerType(), True),
            StructField("x_flag", IntegerType(), True),
            StructField("y_flag", IntegerType(), True),
        ]
    )

    data = [
        ("Gendry", pd.Timestamp("2012-05-01").date(), 15461, 1, 0),
        ("Gendry", pd.Timestamp("2012-05-02").date(), 15462, 0, None),
        ("Gendry", pd.Timestamp("2012-05-03").date(), 15463, 1, None),
        ("Gendry", pd.Timestamp("2012-05-04").date(), 15464, 0, 1),
        ("Gendry", pd.Timestamp("2012-05-05").date(), 15465, 1, None),
        ("Arya", pd.Timestamp("2012-05-06").date(), 15466, 1, 0),
        ("Arya", pd.Timestamp("2012-05-07").date(), 15467, 0, None),
        ("Arya", pd.Timestamp("2012-05-08").date(), 15468, 0, 1),
        ("Arya", pd.Timestamp("2012-05-09").date(), 15469, 1, None),
        ("Arya", pd.Timestamp("2012-05-10").date(), 15470, 1, 2),
        ("Cersei", pd.Timestamp("2012-05-10").date(), 15470, 0, 0),
        ("Cersei", pd.Timestamp("2012-05-11").date(), 15471, 1, None),
        ("Cersei", pd.Timestamp("2012-05-12").date(), 15472, 1, None),
        ("Cersei", pd.Timestamp("2012-05-13").date(), 15473, 1, None),
        ("Cersei", pd.Timestamp("2012-05-15").date(), 15475, 0, 1),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def mock_customers_df(spark):
    """Mock customers df."""
    data = [
        ["bla-1"],
        ["bla-2"],
        ["bla-3"],
    ]

    schema = StructType([StructField("customer_id", StringType())])

    df = spark.createDataFrame(data, schema)

    return df


@pytest.fixture
def mock_spine_df(spark):
    """Mock spine df."""
    data = [
        ["bla-1", datetime.date(2019, 11, 30)],
        ["bla-2", datetime.date(2019, 11, 30)],
        ["bla-3", datetime.date(2019, 11, 30)],
        ["bla-1", datetime.date(2019, 12, 31)],
        ["bla-2", datetime.date(2019, 12, 31)],
        ["bla-3", datetime.date(2019, 12, 31)],
    ]

    schema = StructType(
        [
            StructField("element_id", StringType()),
            StructField("observation_dt", DateType()),
        ]
    )

    df = spark.createDataFrame(data, schema)

    return df


@pytest.fixture
def mock_interactions_df(spark):
    """Mock interactions df."""
    data = [
        [1, "bla-1", datetime.date(2019, 11, 5), "1", "asp"],
        [2, "bla-2", datetime.date(2019, 11, 6), "1", "asp"],
        [3, "bla-2", datetime.date(2019, 11, 7), "1", "asp"],
        [4, "bla-2", datetime.date(2019, 11, 8), "2", "pan"],
        [5, "bla-3", datetime.date(2019, 12, 10), "1", "asp"],
        [6, "bla-3", datetime.date(2019, 12, 11), "1", "pan"],
        [7, "bla-3", datetime.date(2019, 12, 15), "1", "pan"],
        [8, "bla-3", datetime.date(2019, 12, 23), "1", "asp"],
    ]

    schema = StructType(
        [
            StructField("interaction_id", StringType()),
            StructField("customer_id", StringType()),
            StructField("interaction_dt", DateType()),
            StructField("channel_cd", StringType()),
            StructField("product_cd", StringType()),
        ]
    )

    df = spark.createDataFrame(data, schema).withColumn(
        "interaction_last_dt", f.last_day(f.col("interaction_dt"))
    )

    return df


@pytest.fixture
# fmt: off
def mock_key_messages_df(spark):
    """Mock key messages df."""
    data = [
        [1, "bla"], [1, "bla2"], [1, "bla3"],
        [2, "bla"],
        [3, "bla"], [3, "bla3"], [3, "bla5"],
        [4, "bla"],
        [5, "bla2"], [5, "bla7"],
        [6, "bla"], [6, "bla7"],
        [7, "bla"],
        [8, "bla7"],
    ]

    schema = StructType([
        StructField('interaction_id', StringType()),
        StructField('key_message_cd', StringType()),
    ])

    df = spark.createDataFrame(data, schema)

    return df


@pytest.fixture
def mock_numbers_df(spark):
    """Mock numbers df."""
    data = [
        ["id_1", 1, 0.0],
        ["id_2", None, 2.3],
        ["id_3", 10, -2.1],
        ["id_4", 5, None],
        ["id_5", 0, 2.2],
        ["id_6", None, 20.5],
    ]

    schema = StructType(
        [
            StructField("id", StringType()),
            StructField("num1", IntegerType()),
            StructField("num2", DoubleType()),
        ]
    )

    df = spark.createDataFrame(data, schema)

    return df


@pytest.fixture
def mock_event_df(spark):
    """Mock array df."""
    schema = StructType(
        [
            StructField("date_index", IntegerType(), True),
            StructField("person", StringType(), True),
            StructField(
                "tags",
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
        (1, "Jon", []),
        (2, "Jon", [("had_fever", 1.0), ("number", 2.5)]),
        (3, "Jon", [("visit_gp", 1.0), ("number", 3.0), ("bogus", 5.0)]),
        (4, "Jon", [("had_tummy_ache", 1.0), ("number", 4.0)]),
        (5, "Jon", [("had_fever", 1.0), ("number", 1.0)]),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def mock_range_df(spark):
    """Mock range df."""
    schema = StructType(
        [
            StructField("person", StringType(), True),
            StructField("int_start", IntegerType(), True),
            StructField("int_end", IntegerType(), True),
            StructField("date_start", DateType()),
            StructField("date_end", DateType()),
        ]
    )

    data = [
        ("a", 1, 5, datetime.date(2020, 1, 15), datetime.date(2020, 1, 20)),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def mock_before_arrays(spark):
    """Mock before array df."""
    schema = StructType(
        [
            StructField("person", StringType(), True),
            StructField("time_index", IntegerType(), True),
            StructField("value", IntegerType(), True),
        ]
    )

    data = [
        ("a", 1, 10),
        ("a", 3, 8),
        ("a", 5, 6),
        ("a", 2, 9),
        ("a", 4, 7),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def mock_arrays_df(spark):
    """Mock array df."""
    schema = StructType(
        [
            StructField("person", StringType(), True),
            StructField("time_index", ArrayType(DoubleType()), True),
            StructField("value", ArrayType(DoubleType()), True),
        ]
    )

    data = [("a", [1.0, 2.0, 3.0, 4.0, 6.0], [1.0, 2.0, 1.0, 3.0, 1.0])]

    df = spark.createDataFrame(data, schema)
    df = df.withColumn("zipped", f.arrays_zip("time_index", "value"))

    return df


@pytest.fixture
def mock_arrays_df2(spark):
    """Mock array df."""
    schema = StructType(
        [
            StructField("person", StringType(), True),
            StructField("time_index", ArrayType(IntegerType()), True),
            StructField("value", ArrayType(IntegerType()), True),
            StructField("lower_bound", IntegerType(), True),
            StructField("upper_bound", IntegerType(), True),
        ]
    )

    data = [
        (
            "a",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            3,
            5,
        ),
        (
            "a",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            -1,
            1,
        ),
        (
            "a",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            -3,
            3,
        ),
        (
            "a",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            -1,
            -5,
        ),
    ]

    df = spark.createDataFrame(data, schema)
    df = df.withColumn("zipped", f.arrays_zip("time_index", "value"))

    return df


@pytest.fixture
def mock_arrays_df3(spark):
    """Mock array df."""
    schema = StructType(
        [
            StructField("person", StringType(), True),
            StructField("time_index", IntegerType(), True),
            StructField("value", IntegerType(), True),
            StructField("lower_bound", IntegerType(), True),
            StructField("upper_bound", IntegerType(), True),
        ]
    )

    data = [
        ("a", 1, 10, 3, 5),
        ("a", 3, 8, -1, 1),
        ("a", 5, 6, -3, 3),
        ("a", 2, 9, -5, -1),
        ("a", 4, 7, 0, 0),
    ]

    df = spark.createDataFrame(data, schema)

    w = (
        Window.partitionBy("person")
        .orderBy("time_index")
        .rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )
    df = df.withColumn(
        "entire_window", f.collect_list(f.struct("time_index", "value")).over(w)
    )

    return df


@pytest.fixture(scope="module")
def mock_tag_df(spark):
    """Mock sample df."""
    schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("date_index", IntegerType(), True),
            StructField("date", StringType(), True),
            StructField("val", IntegerType(), True),
        ]
    )

    data = [
        ("batch_0001", 1, "2018-01-01 08:42:09", 13),
        ("batch_0001", 2, "2018-01-02 03:24:24", 9),
        ("batch_0001", 3, "2018-01-01 16:00:17", 1),
        ("batch_0001", 4, "2018-01-01 03:40:08", 10),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_data_agg_df():
    """Mock sample df for aggregation."""
    pd_df = pd.DataFrame(
        [
            {
                "pred_entity_id": 1,
                "source": "aact",
                "site_id": "aact-1",
                "site_name": "University Hospital",
                "site_address": "100 street",
                "site_num": 1,
            },
            {
                "pred_entity_id": 1,
                "source": "informa",
                "site_id": "informa-1",
                "site_name": "Respiratory Department of Hospital University",
                "site_address": "100 st",
                "site_num": 2,
            },
            {
                "pred_entity_id": 1,
                "source": "informa",
                "site_id": "informa-2",
                "site_name": "Unversity Hospital Oncology",
                "site_address": "100 st",
                "site_num": 3,
            },
            {
                "pred_entity_id": 1,
                "source": "internal",
                "site_id": "internal-1",
                "site_name": None,
                "site_address": "One Hundred Street",
                "site_num": 4,
            },
            {
                "pred_entity_id": 1,
                "source": "internal",
                "site_id": "internal-5",
                "site_name": "University Hospital Dept. of Oncology",
                "site_address": "One Hundred Street",
                "site_num": 11,
            },
            {
                "pred_entity_id": 1,
                "source": "internal",
                "site_id": "internal-7",
                "site_name": "University Hospital Department of Respiratory",
                "site_address": "One Hundred Street",
                "site_num": 12,
            },
            {
                "pred_entity_id": 2,
                "source": "aact",
                "site_id": "aact-2",
                "site_name": "Hosp Queen Eli",
                "site_address": "Second Street",
                "site_num": 5,
            },
            {
                "pred_entity_id": 2,
                "source": "informa",
                "site_id": "informa-6",
                "site_name": "Hospital Queen Elizabeth",
                "site_address": "2nd St",
                "site_num": 6,
            },
            {
                "pred_entity_id": 2,
                "source": "informa",
                "site_id": "informa-5",
                "site_name": "Hospital Queen Elizabeth",
                "site_address": "2nd Street",
                "site_num": 7,
            },
        ]
    )
    spark = SparkSession.builder.config("spark.sql.shuffle.partitions", 1).getOrCreate()
    df = spark.createDataFrame(pd_df)

    return df


@pytest.fixture(scope="module")
def event_tag_df():
    """Mock event tag dataframe."""
    spark = SparkSession.builder.config("spark.sql.shuffle.partitions", 1).getOrCreate()

    schema = StructType(
        [
            StructField("person_id", StringType()),
            StructField("date_day", DateType()),
            StructField("date_index", LongType()),
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
            "person_1",
            datetime.date(2020, 1, 1),
            18262,
            [("dx_fever", 1.0), ("dx_cancer", 1.0)],
        ),
        ("person_1", datetime.date(2020, 2, 1), 18293, [("dx_fever", 1.0)]),
        (
            "person_1",
            datetime.date(2020, 3, 1),
            18322,
            [("dx_fever", 1.0), ("dx_cold", 1.0)],
        ),
    ]

    return spark.createDataFrame(data, schema=schema)


@pytest.fixture
def df_sample_to_create_tags(spark):
    """Sample df."""
    schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("occupation", StringType(), True),
            StructField("house", ArrayType(StringType()), True),
            StructField("number", ArrayType(IntegerType()), True),
        ]
    )

    data = [
        ("Gendry", 31, "Data Engineer", ["House Baratheon"], [1, 2, 3]),
        ("Jaime", 12, "Data Scientist", ["House Lannister"], [2, 3, 4]),
        ("Tyrion", 65, "Data Analyst", ["House Lannister", "House Stark"], [3, 4, 5]),
        ("Cersei", 29, "Engagement Manager", ["House Lannister"], [5, 6, 7]),
        ("Jon", 31, "Software Engineer", ["House Targaryen", "House Stark"], [6, 7, 8]),
        ("Arya", 27, "MLE", ["House Stark"], [7, 8, 9]),
        (
            "Sansa",
            26,
            "daata translator",
            ["House Stark", "House Lannister"],
            [8, 9, 10],
        ),
        ("Daenerys", 36, "Mother of Dragons", ["House Targaryen"], [9, 10, 11]),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture()
def df_sample_with_tags(spark):
    """Sample df."""
    schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("occupation", StringType(), True),
            StructField("house", ArrayType(StringType()), True),
            StructField("number", ArrayType(IntegerType()), True),
            StructField(
                "tags",
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
            "Gendry",
            31,
            "Data Engineer",
            ["House Baratheon"],
            [1, 2, 3],
            [("occupation-data", 1.0)],
        ),
        (
            "Jaime",
            12,
            "Data Scientist",
            ["House Lannister"],
            [2, 3, 4],
            [("occupation-data", 1.0)],
        ),
        (
            "Tyrion",
            65,
            "Data Analyst",
            ["House Lannister", "House Stark"],
            [3, 4, 5],
            [("occupation-data", 1.0)],
        ),
        (
            "Cersei",
            29,
            "Engagement Manager",
            ["House Lannister"],
            [5, 6, 7],
            [("occupation-not-data", 1.0)],
        ),
        (
            "Jon",
            31,
            "Software Engineer",
            ["House Targaryen", "House Stark"],
            [6, 7, 8],
            [("extra-tag", 1.0)],
        ),
        (
            "Arya",
            27,
            "MLE",
            ["House Stark"],
            [7, 8, 9],
            [("occupation-not-data", 1.0)],
        ),
        (
            "Sansa",
            26,
            "daata translator",
            ["House Stark", "House Lannister"],
            [8, 9, 10],
            [("occupation-data", 1.0), ("extra-tag", 1.0)],
        ),
        (
            "Daenerys",
            36,
            "Mother of Dragons",
            ["House Targaryen"],
            [9, 10, 11],
            [("occupation-not-data", 1.0)],
        ),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture()
def df_sample_with_extracted_tags(spark):
    """Sample df with tags that have been extracted."""
    schema = StructType(
        [
            StructField(
                "tag-xyz",
                StructType(
                    [
                        StructField("tag", StringType(), True),
                        StructField("value", DoubleType(), True),
                    ]
                ),
            ),
            StructField(
                "tag-abc",
                StructType(
                    [
                        StructField("tag", StringType(), True),
                        StructField("value", DoubleType(), True),
                    ]
                ),
            ),
        ]
    )

    data = [
        (
            ("tag-abc", 1.0),
            ("tag-xyz", 2.1),
        ),
        (
            None,
            ("tag-xyz", 3.7),
        ),
        (
            ("tag-abc", 0.0),
            None,
        ),
        (
            ("tag-abc", -1.0),
            ("tag-xyz", 101.2),
        ),
        (None, None),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture(scope="module")
def _get_sample_spark_data_frame3(spark):
    """Mock sample df."""
    schema = StructType(
        [
            StructField("npi_id", StringType(), True),
            StructField("observation_dt", DateType(), True),
            StructField("date_index", IntegerType(), True),
            StructField("all_patient", ArrayType(StringType()), True),
            StructField("male_patient", ArrayType(StringType()), True),
            StructField("female_patient", ArrayType(StringType()), True),
        ]
    )

    data = [
        (
            "0000001738",
            pd.Timestamp("2018-8-31").date(),
            17774,
            ["pt_0000741"],
            [],
            ["pt_0000741"],
        ),
        (
            "0000001738",
            pd.Timestamp("2018-12-31").date(),
            17896,
            ["pt_0000676"],
            [],
            [],
        ),
        (
            "0000001546",
            pd.Timestamp("2019-2-28").date(),
            17955,
            ["pt_0001061"],
            ["pt_0001061"],
            [],
        ),
        (
            "0000001546",
            pd.Timestamp("2019-9-30").date(),
            18169,
            ["pt_0003713"],
            [],
            ["pt_0003713"],
        ),
        (
            "0000001277",
            pd.Timestamp("2018-9-30").date(),
            17804,
            ["pt_0004799"],
            ["pt_0004799"],
            [],
        ),
        (
            "0000001277",
            pd.Timestamp("2020-3-31").date(),
            18352,
            ["pt_0002642"],
            [],
            ["pt_0002642"],
        ),
        (
            "0000001542",
            pd.Timestamp("2018-10-31").date(),
            17835,
            ["pt_0002404"],
            ["pt_0002404"],
            [],
        ),
        (
            "0000001542",
            pd.Timestamp("2020-5-31").date(),
            18413,
            ["pt_0004794"],
            [],
            ["pt_0004794"],
        ),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture(scope="module")
def _get_sample_spark_data_frame4(spark):
    """Mock sample df."""
    schema = StructType(
        [
            StructField("npi_id", StringType(), True),
            StructField("observation_dt", DateType(), True),
            StructField("week_index", IntegerType(), True),
            StructField("feature", ArrayType(IntegerType()), True),
        ]
    )

    data = [
        (
            "1",
            pd.Timestamp("2023-01-01").date(),
            1,
            [12, 11, 8],
        ),
        (
            "1",
            pd.Timestamp("2023-01-08").date(),
            2,
            [33, 29],
        ),
        (
            "1",
            pd.Timestamp("2023-01-15").date(),
            3,
            [44, 39, 35, 74],
        ),
        (
            "2",
            pd.Timestamp("2023-01-01").date(),
            1,
            [14, 13],
        ),
        (
            "2",
            pd.Timestamp("2023-01-08").date(),
            2,
            [5, 10],
        ),
        (
            "2",
            pd.Timestamp("2023-01-15").date(),
            3,
            [44],
        ),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def df_input_slice(spark):
    """Mock sample df."""
    return spark.createDataFrame(
        [
            Row(
                npi_id="h1",
                time_index=1,
                value=0,
            ),
            Row(
                npi_id="h1",
                time_index=2,
                value=1,
            ),
            Row(
                npi_id="h1",
                time_index=3,
                value=2,
            ),
            Row(
                npi_id="h1",
                time_index=4,
                value=3,
            ),
            Row(
                npi_id="h1",
                time_index=5,
                value=4,
            ),
            Row(
                npi_id="h1",
                time_index=6,
                value=5,
            ),
            Row(
                npi_id="h1",
                time_index=7,
                value=6,
            ),
            Row(
                npi_id="h1",
                time_index=8,
                value=7,
            ),
            Row(
                npi_id="h1",
                time_index=9,
                value=8,
            ),
            Row(
                npi_id="h1",
                time_index=10,
                value=9,
            ),
            Row(
                npi_id="h1",
                time_index=11,
                value=10,
            ),
        ]
    )


@pytest.fixture
def mock_strings_df(spark):
    """Mock strings df."""
    data = [
        ["GA90", "10"],
        ["mn78", "23"],
        ["12TH", "33"],
        ["-WE2", None],
        ["+st2", "28"],
        ["TH60", "X"],
    ]

    schema = StructType(
        [
            StructField("sample_col_1", StringType()),
            StructField("sample_col_2", StringType()),
        ]
    )

    df = spark.createDataFrame(data, schema)

    return df
