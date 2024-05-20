from kedro.pipeline import Pipeline, node, pipeline

from .nodes_primary import (
    reduce_join_last_n_days,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=reduce_join_last_n_days,
                inputs=[
                    "intermediate.prints",
                    "intermediate.taps",
                    "intermediate.pays",
                ],
                outputs=["primary_joined.mdt"],
                name="primary_join_tables_node",
            ),
        ]
    )
