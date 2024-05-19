"""Project pipelines."""
from typing import Dict

import meli.pipelines.data_processing.data_ingestion as di
import meli.pipelines.data_processing.data_engineering.intermediate as intermediate

from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ingestion_historical_pipeline = di.create_pipeline()
    intermediate_pipeline =  intermediate.create_pipeline()
    de_historical_pipeline = ingestion_historical_pipeline + intermediate_pipeline

    # TODO: "incremental_de_pipeline"

    return {
        "__default__": de_historical_pipeline,
        "ingestion_historical_pipeline": ingestion_historical_pipeline,
        "intermediate_pipeline": intermediate_pipeline,
        "de_historical_pipeline": de_historical_pipeline
    }
