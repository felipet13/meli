"""Join methods rewritten as functions."""

from datetime import timedelta
from logging import getLogger
from typing import Dict

from dateutil.relativedelta import relativedelta
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, max

logger = getLogger(__name__)


def calculate_target_weekday_date(date, target_weekday, weeks_prior):
    # Subtract weeks_prior from the given date
    target_date = date - relativedelta(weeks=weeks_prior)

    # Adjust the date to the target weekday
    if target_date.weekday() != target_weekday:
        if target_weekday == 0:  # Monday
            target_date = target_date - timedelta(days=target_date.weekday())
        elif target_weekday == 6:  # Sunday
            days_to_add = 6 - target_date.weekday()
            target_date = target_date + timedelta(days=days_to_add)

    return target_date


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


def reduce_join_last_n_days(
    prints_df: SparkDataFrame,
    taps_df: SparkDataFrame,
    pays_df: SparkDataFrame,
    # parameters: Dict,
) -> SparkDataFrame:
    """Joins a list of dataframes into a single dataframe on the specified join_key taking into consideration number of days to load in memory.

    Args:
        list_of_dataframes: A list of dataframes
        join_key: The join key for all the dataframes in ``list_of_dataframes``
        how: Spark SQL join type.

    Returns:
        A spark dataframe.
    """
    # Retrieve last available date in DataFrame using aggregate pushdown

    # Push down filtering to source so only last month data is loaded
    breakpoint()
    spark = pays_df.sparkSession
    spark.sparkContext.setJobDescription("test")
    pays_df.agg(max("pay_date")).collect()[0][0]

    # last 4 complete weeks, usar offset de 1 dia o algo
    # taps_df = taps_df.filter(max(taps_df.date))
    # last_4_weeks = df.filter((df('Date') > date_add(current_timestamp(), 5)).select("Event_Time","User_ID","Impressions","Clicks","URL", "Date")

    # ).first()[0]

    # joined_df = prints_df.join(
    #     taps_df, on=parameters["join_key"], how=parameters["how"]
    # ).join(pays_df, on=parameters["join_key"], how=parameters["how"])
    pass

    return
