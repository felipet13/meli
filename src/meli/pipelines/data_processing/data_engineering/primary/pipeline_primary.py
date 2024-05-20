from kedro.pipeline import Pipeline, node, pipeline

from .nodes_primary import (
    load_last_4_weeks,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_last_4_weeks,
                inputs=[
                    "intermediate.pays",
                    "params:primary_pays",
                ],
                outputs=["primary_loaded.pays"],
                name="primary_load_pays_node",
            ),
            # node(
            #     func=reduce_join_last_n_days,
            #     inputs=[
            #         "intermediate.prints",
            #         "intermediate.taps",
            #         "intermediate.pays",
            #     ],
            #     outputs=["primary_joined.mdt"],
            #     name="primary_join_tables_node",
            # ),
        ]
    )
