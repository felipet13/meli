"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

import meli.pipelines.data_processing.data_ingestion as di
from meli.pipelines.data_processing.data_engineering import intermediate, primary


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ingestion_historical_pipeline = di.create_pipeline()
    intermediate_pipeline = intermediate.create_pipeline()
    primary_pipeline = primary.create_pipeline()
    de_historical_pipeline = (
        ingestion_historical_pipeline + intermediate_pipeline + primary_pipeline
    )

    # TODO: "incremental_de_pipeline"

    return {
        "__default__": de_historical_pipeline,
        "ingestion_historical_pipeline": ingestion_historical_pipeline,
        "intermediate_pipeline": intermediate_pipeline,
        "de_historical_pipeline": de_historical_pipeline,
        "primary_pipeline": primary_pipeline,
    }
