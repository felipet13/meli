"""Project pipelines."""
from typing import Dict

import meli.pipelines.data_processing.data_ingestion as di
import meli.pipelines.data_processing as de

from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_ingestion_pipeline = di.create_pipeline()
    data_engineering_pipeline = de.create_pipeline()

    return {
        "__default__": data_ingestion_pipeline + data_engineering_pipeline,
        "data_ingestion": data_ingestion_pipeline,
        "data_engineering": data_engineering_pipeline,
    }
