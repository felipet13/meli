from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_json_df,

)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_json_df,
                inputs=["prints", "params:raw_prints"],
                outputs=["preprocessed_prints", "prints_columns"],
                name="preprocess_prints_node",
            ),
            node(
                func=preprocess_json_df,
                inputs=["taps", "params:raw_taps"],
                outputs=["preprocessed_taps", "taps_columns"],
                name="preprocess_taps_node",
            ),
        ]
    )
