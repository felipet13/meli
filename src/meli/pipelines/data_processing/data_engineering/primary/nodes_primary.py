"""Join methods rewritten as functions."""

from logging import getLogger
from typing import Dict

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, max, when

from .utils.calculate_date import calculate_target_weekday_date

logger = getLogger(__name__)


def load_last_4_weeks(
    df: SparkDataFrame,
    parameters: Dict,
) -> SparkDataFrame:
    """Loads in memory using a predicate push down filter to source only the last 4 complete weeks of data.
    Args:
        df: dataframe to be filtered.
        parameters: Parameters defined in parameters/data_processing.yml.

    Returns:
        A spark dataframe.
    """
    # Retrieve last available date in DataFrame using aggregate pushdown boosting speed
    max_date = df.select(max(parameters["date_col"])).collect()[0][0]

    # Calculate the date of the Monday 4 weeks prior to the max date
    starting_monday_date = calculate_target_weekday_date(max_date, 0, 4)

    # Calculate the date of the Sunday of the week prior to the max date
    sunday_week_prior = calculate_target_weekday_date(max_date, 6, 1)

    # log the dates with logger info to console
    logger.info(f"Starting Monday Date: {starting_monday_date}")
    logger.info(f"Sunday Week Prior: {sunday_week_prior}")

    # Push down filtering to source so only last month data partitions are loaded
    # filter df column pay_date using 2 dates
    filter_condition = col(parameters["date_col"]).between(
        str(starting_monday_date), str(sunday_week_prior)
    )
    df = df.filter(filter_condition)

    return df


def join_dataframes(
    prints_df: SparkDataFrame,
    taps_df: SparkDataFrame,
    pays_df: SparkDataFrame,
    parameters: Dict,
) -> SparkDataFrame:
    """Joins the dataframes in ``list_of_dataframes`` on the join key.

    Args:
        list_of_dataframes: A list of dataframes
        join_key: The join key for all the dataframes in ``list_of_dataframes``
        how: Spark SQL join type.

    Returns:
        A spark dataframe.
    """
    join_conditions_1 = [
        prints_df[col1] == taps_df[col2]
        for col1, col2 in zip(
            parameters["prints_df"]["cols_to_join"],
            parameters["taps_df"]["cols_to_join"],
        )
    ]

    joined_df_1 = (
        prints_df.join(taps_df, on=join_conditions_1, how=parameters["how"])
        .select(
            prints_df["date"],
            prints_df["position"],
            prints_df["value_prop"],
            prints_df["user_id"],
            prints_df["year"],
            prints_df["week_of_year"],
            taps_df["date"].alias("date_taps"),
        )
        .withColumn("customer_tap", when(col("date_taps").isNull(), 0).otherwise(1))
        .drop("date_taps")
    )

    join_conditions_2 = [
        joined_df_1["date"] == pays_df["pay_date"],
        joined_df_1["user_id"] == pays_df["user_id"],
        joined_df_1["value_prop"] == pays_df["value_prop"],
    ]

    joined_df_final = (
        joined_df_1.join(pays_df, on=join_conditions_2, how=parameters["how"])
        .select(
            joined_df_1["date"],
            joined_df_1["position"],
            joined_df_1["value_prop"],
            joined_df_1["user_id"],
            joined_df_1["year"],
            joined_df_1["week_of_year"],
            joined_df_1["customer_tap"],
            pays_df["total"].alias("paid_by_customer"),
        )
        .fillna(0)
    )

    return joined_df_final
