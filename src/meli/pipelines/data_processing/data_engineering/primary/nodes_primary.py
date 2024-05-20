"""Join methods rewritten as functions."""

from datetime import timedelta
from typing import Dict

from dateutil.relativedelta import relativedelta
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, max


def calculate_four_weeks_prior_monday_date(date):
    # Subtract 4 weeks from the given date
    four_weeks_prior = date - relativedelta(weeks=4)

    # Adjust the date to the nearest Monday
    if four_weeks_prior.weekday() != 0:
        four_weeks_prior = four_weeks_prior - timedelta(days=four_weeks_prior.weekday())

    return four_weeks_prior


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
    starting_monday_date = calculate_four_weeks_prior_monday_date(max_date)

    sunday_week_prior = max_date - relativedelta(weeks=1)

    if sunday_week_prior.weekday() != 6:
        # Calculate the number of days to add to get to Sunday
        days_to_add = 6 - sunday_week_prior.weekday()
        sunday_week_prior = sunday_week_prior + timedelta(days=days_to_add)

    # Push down filtering to source so only last month data partitions are loaded
    # filter df column pay_date using 2 dates
    df = df.filter(
        col(parameters["date_col"]).between(
            str(starting_monday_date), str(sunday_week_prior)
        )
    )

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
