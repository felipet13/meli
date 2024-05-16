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

from feature_generation.v1.core.utils.join import reduce_join


def test_reduce_join(get_sample_spark_data_frame):
    list_of_df = [
        get_sample_spark_data_frame.select("name"),
        get_sample_spark_data_frame.select("name"),
        get_sample_spark_data_frame.select(f.substring("name", 0, 1).alias("name")),
    ]

    join_of_all = reduce_join(list_of_df, join_key="name", how="outer")

    assert join_of_all.count() == 2 * list_of_df[0].count()
