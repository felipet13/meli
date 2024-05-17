from kedro.pipeline import Pipeline, node, pipeline

from meli.pipelines.data_processing.nodes import preprocess_json_df

from .nodes_data_ingestion import ingest_historical_data


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
        ]
    )
