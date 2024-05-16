# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.

"""Contains date and datetime related utility functions."""

import re
from typing import Union

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Column

from ..utils.alias import alias


@alias()
def date_index(
    column: Union[str, pyspark.sql.Column], reference_date: str = "1970-01-01"
) -> Column:
    """Creates an integer column given a date representing days.

    Difference in index represents difference in number of days. Useful for window
    functions using rangeBetween.

    Use `alias` argument for setting the name of the output column.

    Args:
        column: The name of the date column to be converted to an integer.
        reference_date: Date to be compared against. Defaults to 1970-01-01.

    Returns:
        A spark column object.

    Raises:
        ValueError: If datetime does not conform to the pattern yyyy-mm-dd
    """
    if not re.match(r"\d{4}-\d{2}-\d{2}", reference_date):
        raise ValueError("Date format should follow yyyy-mm-dd.")

    date_col = column if isinstance(column, pyspark.sql.Column) else f.col(column)

    return f.datediff(date_col, f.lit(reference_date).cast("date"))


@alias()
def month_index(
    column: Union[str, pyspark.sql.Column], reference_date: str = "1970-01-01"
) -> Column:
    """Creates an integer column given a date representing months.

    Difference in index represents difference in number of months. Useful for window
    functions using rangeBetween.

    Use `alias` argument for setting the name of the output column.

    Args:
        column: The name of the date column to be converted to an integer.
        reference_date: Date to be compared against. Defaults to 1970-01-01.

    Returns:
        A spark column object

    Raises:
        ValueError: If datetime does not conform to the pattern yyyy-mm-dd
    """
    if not re.match(r"\d{4}-\d{2}-\d{2}", reference_date):
        raise ValueError("Date format should follow yyyy-mm-dd.")

    date_col = column if isinstance(column, pyspark.sql.Column) else f.col(column)

    return f.months_between(date_col, f.lit(reference_date).cast("date")).cast("int")


@alias()
def quarter_index(
    column: Union[str, pyspark.sql.Column], reference_date: str = "1970-01-01"
) -> Column:
    """Creates an integer column given a date representing quarters.

    Difference in index represents difference in number of quarters. Useful for window
    functions using rangeBetween.

    Use `alias` argument for setting the name of the output column.

    Args:
        column: The name of the date column to be converted to an integer.
        reference_date: Date to be compared against. Defaults to 1970-01-01.

    Returns:
        A spark column object

    Raises:
        ValueError: If datetime does not conform to the pattern yyyy-mm-dd
    """
    if not re.match(r"\d{4}-\d{2}-\d{2}", reference_date):
        raise ValueError("Date format should follow yyyy-mm-dd.")

    date_col = column if isinstance(column, pyspark.sql.Column) else f.col(column)

    return f.months_between(date_col, f.lit(reference_date).cast("date")) / f.lit(3)


@alias()
def hour_index(
    column: Union[str, pyspark.sql.Column], reference_ts: str = "1970-01-01 00:00:00"
) -> Column:
    """Creates a time index column of type double representing hours, given a timestamp.

    Difference in index represents difference in number of hours. Useful for window
    functions using rangeBetween.

    Use `alias` argument for setting the name of the output column.

    Args:
        column: The name of the timestamp column to be converted to a double.
        reference_ts: Timestamp to be compared against. Defaults to
            `1970-01-01 00:00:00`.

    Returns:
        A spark column object containing a time index in an hour format.

    Raises:
        ValueError: If datetime does not confirm to the pattern `yyyy-mm-dd HH:mm:ss`.

    """
    if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", reference_ts):
        raise ValueError("Date format should follow yyyy-mm-dd HH:mm:ss")

    time_col = column if isinstance(column, pyspark.sql.Column) else f.col(column)

    return (
        time_col.cast("long") - f.lit(reference_ts).cast("timestamp").cast("long")
    ) / 3600


@alias()
def minute_index(
    column: Union[str, pyspark.sql.Column], reference_ts: str = "1970-01-01 00:00:00"
) -> Column:
    """Creates a time index column of type double representing minutes given timestamp.

    Difference in index represents difference in number of hours. Useful for window
    functions using rangeBetween.

    Use `alias` argument for setting the name of the output column.

    Args:
        column: The name of the timestamp column to be converted to a double.
        reference_ts: Timestamp to be compared against. Defaults to
            `1970-01-01 00:00:00`.

    Returns:
        A spark column object containing a time index in an hour format.

    Raises:
        ValueError: If datetime does not confirm to the pattern `yyyy-mm-dd HH:mm:ss`.

    """
    if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", reference_ts):
        raise ValueError("Date format should follow yyyy-mm-dd HH:mm:ss")

    time_col = column if isinstance(column, pyspark.sql.Column) else f.col(column)

    return (
        (time_col.cast("long") - f.lit(reference_ts).cast("timestamp").cast("long"))
        / 60
    ).cast("int")


@alias()
def time_add(col: str, time: str) -> pyspark.sql.Column:
    """Adds/subtracts user defined time period to timestamp column.

    The function works for date columns as well
    The user needs to specify negative values for subtracting time, i.e. -1 DAYS

    Use `alias` argument for setting the name of the output column.

    Args:
        col: The input timestamp column.
        time: Time period to add/subtract along with the unit of time.
            Available units are YEAR, MONTH, DAY, HOUR, MINUTE, SECOND,
            MILLISECOND, and MICROSECOND. The plural units also work.
            (Eg. 10 minutes or 10 minute or -1 day.)

    Returns:
        Column with specified time added/subtracted.
    """
    return f.col(col) + f.expr(f"INTERVAL {time}")
