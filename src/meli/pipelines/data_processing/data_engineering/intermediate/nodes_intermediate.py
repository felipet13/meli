import logging
from pprint import pformat
from typing import Dict, Tuple

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, weekofyear, year

# from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


def preprocess_df(
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
    logger.info("`input parameters`:\n%s", pformat(parameters))
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

    # Rename date column so partition name is congruent
    cols_to_rename = parameters.get("columns_to_rename")
    if cols_to_rename:
        for key, val in cols_to_rename.items():
            df_raw = df_raw.withColumnRenamed(
                key,
                val,
            )

    # log renamed columns
    logger.info("`renamed cols`: \n %s", pformat(cols_to_rename))

    # Log columns and types to output at final
    logger.info("`output df columns and types`: \n ")

    df_raw.printSchema()

    # Creates columns Year - Week_of_year for partitions
    if parameters.get("date_col") is None:
        raise ValueError("Parameter `date_col` is required and not set. ")
    else:
        df_raw = df_raw.withColumn(
            "week_of_year", weekofyear(parameters.get("date_col"))
        )
        df_raw = df_raw.withColumn("year", year(parameters.get("date_col")))

    logger.info("`Partitioning by`: [`Year`, `Week_of_year`] ")

    # Return DataFrame and dict column of {column_name:type} for tracking
    return df_raw, {"columns": {k[0]: k[1] for k in df_raw.dtypes}}
