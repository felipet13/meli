from kedro.pipeline import Pipeline, node, pipeline

from .nodes_data_ingestion import ingest_historical_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # memory dataset on node output for 'ingest_prints_node'
            node(
                func=ingest_historical_data,
                inputs=["raw.prints", "params:raw_prints"],
                outputs="ingested_raw.prints",
                name="ingest_prints_node",
            ),
            node(
                func=ingest_historical_data,
                inputs=["raw.taps", "params:raw_taps"],
                outputs="ingested_raw.taps",
                name="ingest_taps_node",
            ),
            node(
                func=ingest_historical_data,
                inputs=["raw.pays", "params:raw_pays"],
                outputs="ingested_raw.pays",
                name="ingest_pays_node",
            ),
        ]
    )
