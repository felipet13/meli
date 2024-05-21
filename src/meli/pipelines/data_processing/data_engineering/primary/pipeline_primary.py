from kedro.pipeline import Pipeline, node, pipeline

from .nodes_primary import (
    join_dataframes,
    load_last_4_weeks,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_last_4_weeks,
                inputs=[
                    "intermediate.prints",
                    "params:primary_prints",
                ],
                outputs="primary_loaded.prints",
                name="primary_load_prints_node",
            ),
            node(
                func=load_last_4_weeks,
                inputs=[
                    "intermediate.taps",
                    "params:primary_taps",
                ],
                outputs="primary_loaded.taps",
                name="primary_load_taps_node",
            ),
            node(
                func=load_last_4_weeks,
                inputs=[
                    "intermediate.pays",
                    "params:primary_pays",
                ],
                outputs="primary_loaded.pays",
                name="primary_load_pays_node",
            ),
            node(
                func=join_dataframes,
                inputs=[
                    "primary_loaded.prints",
                    "primary_loaded.taps",
                    "primary_loaded.pays",
                    "params:join_parameters",
                ],
                outputs=["primary_joined.pre_mdt"],
                name="primary_join_tables_node",
            ),
        ]
    )
