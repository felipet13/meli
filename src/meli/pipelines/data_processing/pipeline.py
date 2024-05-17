from kedro.pipeline import Pipeline, node, pipeline

from meli.utils.ingestion.nodes.ingestor import ingest_historical_data, ingest_incremental_data

from .nodes import (
    preprocess_json_df,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # memory dataset on node output for 'ingest_prints_node'
            node(
                func=ingest_historical_data,
                inputs=["prints", "params:raw_prints"],
                outputs="ingested_prints",
                name="ingest_prints_node",
            ),
        #     node(
        #         func=preprocess_json_df,
        #         inputs=["ingested_prints", "params:raw_prints"],
        #         outputs=["preprocessed_prints", "prints_columns"],
        #         name="preprocess_prints_node",
        #     ),
        #     node(
        #         func=preprocess_json_df,
        #         inputs=["taps", "params:raw_taps"],
        #         outputs=["preprocessed_taps", "taps_columns"],
        #         name="preprocess_taps_node",
        #     ),
        # ]
    )
