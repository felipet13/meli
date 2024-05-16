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
"""Test."""

from feature_generation.v1.core.features.create_column import create_columns_from_config
from feature_generation.v1.core.features.flags import isin


def test_create_columns_from_config_with_sequential(spark):
    df = spark.sql("SELECT 1 as c1")

    result = create_columns_from_config(
        df=df,
        column_instructions=[
            isin(input="c1", values=[1], output="new_col"),
            isin(input="new_col", values=[1], output="new_col2"),
        ],
        params_keep_cols=[".*"],
        enable_regex=True,
        sequential=True,
    )

    assert result.columns == ["c1", "new_col", "new_col2"]
