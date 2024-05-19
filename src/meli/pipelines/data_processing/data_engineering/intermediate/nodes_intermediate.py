import logging
from pprint import pformat
from typing import Dict, Tuple

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col

# from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


def preprocess_json_df(
    df_raw: SparkDataFrame, parameters: Dict
) -> Tuple[SparkDataFrame, Dict]:
    """Preprocesses the data for df_raw.

    Args:
        df_raw: Raw json data.
        parameters: Parameters defined in parameters/data_processing.yml.
    Returns:
        Preprocessed data, with `event_data` column splitting/exploding a column of dicts
        to regular columns.
    """
    logger.info(
            "`parameters`:\n%s", pformat(parameters)
        )
    # Use from parameters dict the column of dicts to be exploded
    if parameters.get("explode_column") is not None:
        logger.info("Explode_column: %s", parameters.get("explode_column"))

        col_to_explode = parameters["explode_column"] + ".*"
        cols_less_col_explode = df_raw.columns.copy()
        cols_less_col_explode.remove(parameters["explode_column"])

        # Explode the column of dicts to regular columns
        df_raw = df_raw.select(*cols_less_col_explode, col_to_explode)

    # enforce data types in declared columns
    if parameters.get("columns_to_cast") is not None:
        logger.info("Columns_to_cast: %s", parameters.get("columns_to_cast"))
        for column, type in parameters["columns_to_cast"].items():
            df_raw = df_raw.withColumn(column, col(column).cast(type))

    # Drop columns that aren't required
    if parameters.get("columns_to_drop"):
        df_raw = df_raw.drop(*parameters["columns_to_drop"])

    return df_raw, {"columns": df_raw.columns, "data_type": "df_raw"}
