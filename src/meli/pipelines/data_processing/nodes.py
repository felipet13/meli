from typing import Dict, Tuple
import logging

import pandas as pd
from pyspark.sql import Column
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


def preprocess_json_df(df_raw: SparkDataFrame, parameters: Dict) -> Tuple[SparkDataFrame, Dict]:
    """Preprocesses the data for df_raw.

    Args:
        df_raw: Raw data.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Preprocessed data, with `event_data` column splitting/exploding a column of dicts
        to regular columns.
    """
    # Use from parameters dict the column of dicts to be exploded
    if parameters.get("explode_column") is not None:
        logger.info("Parameters: %s", parameters)

        col_to_explode = parameters["explode_column"] + ".*"
        cols_less_col_explode = df_raw.columns.copy()
        cols_less_col_explode.remove(parameters["explode_column"])

        # Explode the column of dicts to regular columns
        df_raw = df_raw.select(*cols_less_col_explode, col_to_explode)

    # enforce data types in declared columns
    if parameters.get("columns_to_cast") is not None:
            for column, type in parameters["columns_to_cast"].items():
                df_raw = df_raw.withColumn(column, col(column).cast(type))
 
    # Drop columns that aren't required
    if parameters.get("columns_to_drop"):
            df_raw = df_raw.drop(*parameters["columns_to_drop"])


    return df_raw, {"columns": df_raw.columns, "data_type": "df_raw"}


def _is_true(x: Column) -> Column:
    return x == "t"


def _parse_percentage(x: Column) -> Column:
    x = regexp_replace(x, "%", "")
    x = x.cast("float") / 100
    return x


def _parse_money(x: Column) -> Column:
    x = regexp_replace(x, "[$£€]", "")
    x = regexp_replace(x, ",", "")
    x = x.cast(DoubleType())
    return x


def preprocess_companies(companies: SparkDataFrame) -> Tuple[SparkDataFrame, Dict]:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies = companies.withColumn("iata_approved", _is_true(companies.iata_approved))
    companies = companies.withColumn("company_rating", _parse_percentage(companies.company_rating))

    # Drop columns that aren't used for model training
    companies = companies.drop('company_location', 'total_fleet_count')
    return companies, {"columns": companies.columns, "data_type": "companies"}


def load_shuttles_to_csv(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Load shuttles to csv because it's not possible to load excel directly into spark.
    """
    return shuttles


def preprocess_shuttles(shuttles: SparkDataFrame) -> SparkDataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles = shuttles.withColumn("d_check_complete", _is_true(shuttles.d_check_complete))
    shuttles = shuttles.withColumn("moon_clearance_complete", _is_true(shuttles.moon_clearance_complete))
    shuttles = shuttles.withColumn("price", _parse_money(shuttles.price))

    # Drop columns that aren't used for model training
    shuttles = shuttles.drop('shuttle_location', 'engine_type', 'engine_vendor', 'cancellation_policy')
    return shuttles


def preprocess_reviews(reviews: SparkDataFrame) -> SparkDataFrame:
    # Drop columns that aren't used for model training
    reviews = reviews.drop('review_scores_comfort', 'review_scores_amenities', 'review_scores_trip', 'review_scores_crew', 'review_scores_location', 'review_scores_price', 'number_of_reviews', 'reviews_per_month')
    return reviews


def create_model_input_table(
    shuttles: SparkDataFrame, companies: SparkDataFrame, reviews: SparkDataFrame
) -> SparkDataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    # Rename columns to prevent duplicates
    shuttles = shuttles.withColumnRenamed("id", "shuttle_id")
    companies = companies.withColumnRenamed("id", "company_id")

    rated_shuttles = shuttles.join(reviews, "shuttle_id", how="left")
    model_input_table = rated_shuttles.join(companies, "company_id", how="left")
    model_input_table = model_input_table.dropna()
    return model_input_table
