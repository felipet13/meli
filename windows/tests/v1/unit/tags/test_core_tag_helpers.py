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

from pyspark.sql import Window

from feature_generation.v1.core.tags.tag_helpers import (
    days_since_first,
    days_since_last,
)


def test_days_since_last(event_tag_df):
    input_df = event_tag_df
    window = Window.partitionBy("person_id").orderBy("date_day")

    output_df = input_df.select(
        "person_id",
        "date_day",
        *days_since_last(
            tags=["dx_cancer"],
            tag_col="tags_all",
            time_col="date_day",
            windows_spec=window,
        ),
    )
    assert set(output_df.columns) == {
        "person_id",
        "date_day",
        "days_since_last_dx_cancer",
    }
    assert output_df.count() == input_df.count()
    assert [x[0] for x in output_df.select("days_since_last_dx_cancer").collect()] == [
        0,
        31,
        60,
    ]


def test_days_since_first(event_tag_df):
    input_df = event_tag_df
    window = Window.partitionBy("person_id").orderBy("date_day")

    output_df = input_df.select(
        "person_id",
        "date_day",
        *days_since_first(
            tags=["dx_fever"],
            tag_col="tags_all",
            time_col="date_day",
            windows_spec=window,
        ),
    )
    assert set(output_df.columns) == {
        "person_id",
        "date_day",
        "days_since_first_dx_fever",
    }
    assert output_df.count() == input_df.count()
    assert [x[0] for x in output_df.select("days_since_first_dx_fever").collect()] == [
        0,
        31,
        60,
    ]
