from kedro.pipeline import Pipeline, node, pipeline

from .nodes_intermediate import (
    preprocess_json_df,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_json_df,
                inputs=["ingested_prints", "params:pre_process_prints"],
                outputs=["preprocessed_prints", "prints_columns"],
                name="preprocess_prints_node",
            ),
            node(
                func=preprocess_json_df,
                inputs=["taps", "params:pre_process_taps"],
                outputs=["preprocessed_taps", "taps_columns"],
                name="preprocess_taps_node",
            ),
        ]
    )
