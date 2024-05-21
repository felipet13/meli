from datetime import timedelta

from dateutil.relativedelta import relativedelta

MONDAY = 0
SUNDAY = 6


# Function not finally utilized as I refactored the code, still worth saving
def calculate_target_weekday_date(date, target_weekday, weeks_prior):
    # Subtract weeks_prior from the given date
    target_date = date - relativedelta(weeks=weeks_prior)

    # Adjust the date to the target weekday
    if target_date.weekday() != target_weekday:
        if target_weekday == MONDAY:  # Monday
            target_date = target_date - timedelta(days=target_date.weekday())
        elif target_weekday == SUNDAY:  # Sunday
            days_to_add = SUNDAY - target_date.weekday()
            target_date = target_date + timedelta(days=days_to_add)

    return target_date
