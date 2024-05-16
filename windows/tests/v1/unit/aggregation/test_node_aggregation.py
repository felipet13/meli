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
from pyspark.sql.types import StringType

from feature_generation.v1.nodes.aggregation.aggregate import aggregate_attributes


def select_first(list_of_values):
    @f.udf(returnType=StringType())
    def my_udf(list_of_values):
        return str(list_of_values[0])

    return my_udf(list_of_values)


def test_parse_instructions(sample_data_agg_df, tmp_path):
    key_cols = ["pred_entity_id"]

    columns = {
        "min_site_id": {"object": "pyspark.sql.functions.min", "col": "site_id"},
        "site_id": {
            "object": "tests.v1.unit.aggregation.test_node_aggregation.select_first",
            "list_of_values": {
                "object": "pyspark.sql.functions.collect_list",
                "col": "site_id",
            },
        },
    }

    new_df = aggregate_attributes(
        df=sample_data_agg_df,
        key_cols=key_cols,
        column_instructions=columns,
    )

    assert new_df.select("min_site_id").distinct().collect()[0][0] == "aact-1"

    entity_2_site_id = (
        new_df.filter("pred_entity_id == 2").select("site_id").collect()[0][0]
    )
    assert entity_2_site_id == "aact-2"
