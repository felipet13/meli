from kedro.pipeline import Pipeline, node, pipeline

from .nodes_primary import (
    create_windows,
    join_dataframes,
    load_last_3_weeks,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_last_3_weeks,
                inputs=[
                    "intermediate.prints",
                    "params:primary_prints",
                ],
                outputs="primary_loaded.prints",
                name="primary_load_prints_node",
            ),
            node(
                func=load_last_3_weeks,
                inputs=[
                    "intermediate.taps",
                    "params:primary_taps",
                ],
                outputs="primary_loaded.taps",
                name="primary_load_taps_node",
            ),
            node(
                func=load_last_3_weeks,
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
                    "params:primary_join_parameters",
                ],
                outputs="primary_joined.pre_mdt",
                name="primary_join_tables_node",
            ),
            node(
                func=create_windows,
                inputs=[
                    "primary_joined.pre_mdt",
                    "params:primary_windows_parameters",
                ],
                outputs="primary_joined.mdt",
                name="primary_create_windows_node",
            ),
        ]
    )
