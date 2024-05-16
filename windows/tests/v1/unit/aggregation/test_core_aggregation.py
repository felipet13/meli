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
from pyspark.sql.types import StringType
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from feature_generation.v1.core.aggregation.aggregate import aggregate_attributes


def test_parse_instructions(sample_data_agg_df, tmp_path):
    def k_shingles(string: str, k: int = 3):
        return [string[x : x + k] for x in range(len(string) - k + 1)]

    @f.udf(returnType=StringType())
    def cluster_centers(list_of_values):
        new_list_of_values = [k_shingles(x, 3) for x in list_of_values]
        cv = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
        bag_of_words = cv.fit_transform(new_list_of_values)

        X = bag_of_words
        km = KMeans(
            n_clusters=1,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )

        X_dist = km.fit_transform(X)

        # convert output to dataframe to be used easier
        distance_from_centers = pd.DataFrame(
            X_dist.sum(axis=1).round(2), columns=["dist"]
        )
        distance_from_centers["entries"] = list_of_values

        smallest_distance_entry = distance_from_centers.sort_values(["dist"])[
            "entries"
        ].iloc[0]

        return smallest_distance_entry

    @f.udf()
    def array_min(col):
        return f.array_min(col)

    @f.udf(returnType=StringType())
    def get_first(site_name, site_address):
        df = pd.DataFrame(
            {
                "site_name": site_name,
                "site_address": site_address,
            }
        )
        return df.sort_values(["site_name"], ascending=False).site_name.iloc[0]

    @f.udf(returnType=StringType())
    def select_first(list_of_values):
        return str(list_of_values[0])

    key_cols = ["pred_entity_id"]
    columns = {
        "min_site_id": f.min("site_id"),
        "site_id": select_first(f.collect_list("site_id")),
        "site_name": cluster_centers(f.collect_list("site_name")),
        "site_address": select_first(f.collect_list("site_address")),
        "site_num": array_min(f.collect_list("site_num")),
        "test_test": get_first(
            f.collect_list("site_name"),
            f.collect_list("site_address"),
        ),
        "min_site_num": f.min("site_num"),
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

    entity_1_site_name = (
        new_df.filter("pred_entity_id == 1").select("site_name").collect()[0][0]
    )
    assert entity_1_site_name == "University Hospital"

    entity_2_site_name = (
        new_df.filter("pred_entity_id == 2").select("site_name").collect()[0][0]
    )
    assert entity_2_site_name == "Hospital Queen Elizabeth"

    entity_2_site_name = (
        new_df.filter("pred_entity_id == 2").select("min_site_id").collect()[0][0]
    )
    assert entity_2_site_name == "aact-2"

    entity_2_site_num = (
        new_df.filter("pred_entity_id == 2").select("min_site_num").collect()[0][0]
    )
    assert entity_2_site_num == 5
