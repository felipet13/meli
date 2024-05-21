import pytest

from ..core.utils.date_validation import validate_date


def test_valid_date():
    """Test correct dates"""
    dates_correct = ["2022-01-01", "2022-06-30"]
    assert validate_date(dates_correct) is None


def test_invalid_date_format():
    """Test invalid date formats"""
    dates_invalid = ["obi-wan", "kenobi"]
    with pytest.raises(ValueError) as exc_info:
        validate_date(dates_invalid)
        assert type(exc_info.value.__cause__) is ValueError


def test_invalid_date_interval():
    """Test invalid date intervals (end date precedes start)"""
    dates_invalid = ["2022-11-01", "2022-01-01"]
    with pytest.raises(ValueError) as exc_info:
        validate_date(dates_invalid)
        assert type(exc_info.value.__cause__) is ValueError
