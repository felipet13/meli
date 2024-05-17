# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.

# pylint: skip-file
# flake8: noqa
import pytest

from ingestion.core.utils.date_validation import validate_date


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
