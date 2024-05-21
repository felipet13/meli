"""Join methods rewritten as functions."""

from logging import getLogger
from typing import Dict

from dateutil.relativedelta import relativedelta
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, count, max, sum, when
from pyspark.sql.window import Window

logger = getLogger(__name__)


def load_last_3_weeks(
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
    if parameters.get("date_col") is None:
        raise ValueError("date col is missing from parameters")

    # Retrieve last available date in DataFrame using aggregate pushdown for speed boosting, Note is not done in partition column
    # Calculate the max date in the DataFrame
    max_date = df.select(max(parameters["date_col"])).collect()[0][0]

    last_n_days = parameters.get("last_n_days")
    if last_n_days is None:
        logger.warning(
            "last_n_days parameter not found, defaulting to 3 weeks from max date for start date"
        )

        # Calculate the start date of the 3 weeks prior relative to the max date
        starting_day_date = max_date - relativedelta(weeks=3)

    else:
        # If last_n_days parameter present load only offset of start date as end date offset will be implicitly loaded
        starting_day_date = (
            max_date - relativedelta(days=last_n_days) - relativedelta(weeks=3)
        )

    # Calculate the date of the previous day to the max date
    prior_day_date = max_date - relativedelta(days=1)

    # log the dates with logger info to console
    logger.info(f"Starting Day Date: {starting_day_date}")
    logger.info(f"Prior Day Date: {prior_day_date}")

    # Push down partition filtering to source so only necessary data partitions are loaded
    # filter df column `date`` using 2 dates
    filter_condition = col(parameters["date_col"] + "_partition").between(
        str(starting_day_date), str(prior_day_date)
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
            taps_df["date"].alias("date_taps"),
        )
        .withColumn("customer_tap", when(col("date_taps").isNull(), 0).otherwise(1))
        .drop("date_taps")
    )

    # If done as join_conditions_1 generates error TODO investigate
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
            joined_df_1["customer_tap"],
            # pays_df["pay_date"],
            pays_df["total"].alias("paid_by_customer"),
        )
        .fillna(0)
    )
    return joined_df_final


#
def create_windows(df: SparkDataFrame, parameters: Dict):
    """Creates windows for the dataframe.

    Args:
        df: A spark dataframe.
        parameters: Parameters defined in parameters/data_processing.yml.

    Returns:
        A spark dataframe.
    """
    # creates window that calculates the count each user_id has customer_tap in the last 3 weeks prior to the date
    window = (
        Window()
        .partitionBy(["user_id", "value_prop"])
        .orderBy(df[parameters["date_col"]].asc())
        .rowsBetween(-21, -1)
    )
    df = df.withColumn(
        "count_customer_saw_print_last_3_weeks", count("value_prop").over(window)
    )
    df = df.withColumn(
        "count_customer_tap_print_last_3_weeks", sum("customer_tap").over(window)
    )
    df = df.withColumn(
        "sum_customer_paid_last_3_weeks", sum("paid_by_customer").over(window)
    )
    df = df.withColumn(
        "count_customer_paid_last_3_weeks",
        sum(when(col("paid_by_customer") > 0, 1).otherwise(0)).over(window),
    )

    if parameters.get("last_n_days"):
        last_n_days = parameters["last_n_days"]
    else:
        last_n_days = 0

    # filter dataframe using date column for las 7 dates inclusive
    max_date = df.select(max(parameters["date_col"])).collect()[0][0]
    df = df.filter(
        col(parameters["date_col"]).between(
            max_date - relativedelta(days=last_n_days), max_date
        )
    )

    return df, {"columns": {k[0]: k[1] for k in df.dtypes}}
