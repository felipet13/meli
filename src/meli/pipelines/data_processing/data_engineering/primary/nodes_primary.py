"""Join methods rewritten as functions."""

from typing import Dict

from pyspark.sql import DataFrame as SparkDataFrame


def reduce_join_last_n_days(
    prints_df: SparkDataFrame,
    taps_df: SparkDataFrame,
    pays_df: SparkDataFrame,
    parameters: Dict,
) -> SparkDataFrame:
    """Joins a list of dataframes into a single dataframe on the specified join_key taking into consideration number of days to load in memory.

    Args:
        list_of_dataframes: A list of dataframes
        join_key: The join key for all the dataframes in ``list_of_dataframes``
        how: Spark SQL join type.

    Returns:
        A spark dataframe.
    """
    # taps_df = taps_df.filter(taps_df.date )
    #   last_14 = df.filter((df('Date') > date_add(current_timestamp(), -14)).select("Event_Time","User_ID","Impressions","Clicks","URL", "Date")

    # ).first()[0]

    # joined_df = prints_df.join(
    #     taps_df, on=parameters["join_key"], how=parameters["how"]
    # ).join(pays_df, on=parameters["join_key"], how=parameters["how"])
    pass

    return
