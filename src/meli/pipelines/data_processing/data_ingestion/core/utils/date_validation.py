"""Contains date validation functions."""

import datetime
import logging
from typing import Union


def _validate_single_date(date_str: str) -> None:
    """Validates if date is formatted correctly."""
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(
            f"Incorrect date format, should be 'YYYY-MM-DD', got {date_str}"
        ) from e


def _check_valid_interval(dates: Union[list, tuple]) -> None:
    """Check if end date is bigger than start date."""
    start_dt = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(dates[1], "%Y-%m-%d")

    if start_dt > end_dt:
        raise ValueError(
            f"Invalid date interval: start_dt [{start_dt}] is greater than end_date"
            f" [{end_dt}]"
        )


def validate_date(dates: Union[str, list, tuple]) -> None:
    """Validate if dates are formatted correctly.

    The correct format is 'YYYY-MM-DD'.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Date(s) received for validation: %s", dates)

    if isinstance(dates, str):
        _validate_single_date(dates)

    elif type(dates) in [list, tuple]:
        for date in dates:
            _validate_single_date(date)
        _check_valid_interval(dates)

    else:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD.")
