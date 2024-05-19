from kedro.pipeline import Pipeline, node, pipeline

from .nodes_intermediate import (
    preprocess_df,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_df,
                inputs=["ingested_raw.prints", "params:intermediate_prints"],
                outputs=["intermediate.prints", "intermediate.prints_columns"],
                name="intermediate_prints_node",
            ),
            node(
                func=preprocess_df,
                inputs=["ingested_raw.taps", "params:intermediate_taps"],
                outputs=["intermediate.taps", "intermediate.taps_columns"],
                name="intermediate_taps_node",
            ),
        ]
    )
