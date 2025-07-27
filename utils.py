import datetime
import pytz
from datetime import timedelta

# =================================================================
#  Time and Date Utilities
# =================================================================


def datetime_to_timestamp(dt, timezone="US/Pacific"):
    """Convert a datetime object to a timezone-aware timestamp in milliseconds."""
    # If a naive date object is passed, convert it to a datetime at midnight.
    if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        dt = datetime.datetime.combine(dt, datetime.time.min)

    # If the datetime is naive, assume it's UTC.
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = pytz.utc.localize(dt)

    # Convert to the target timezone.
    local_tz = pytz.timezone(timezone)
    local_dt = dt.astimezone(local_tz)

    return int(local_dt.timestamp() * 1000)


def timestamp_to_datetime(timestamp_ms, timezone="US/Pacific"):
    """Convert a timestamp in milliseconds to a timezone-aware datetime object."""
    timestamp_seconds = timestamp_ms / 1000.0
    # Create a timezone-aware UTC datetime directly.
    utc_dt = datetime.datetime.fromtimestamp(timestamp_seconds, tz=datetime.timezone.utc)
    # Convert to the target local timezone.
    return utc_dt.astimezone(pytz.timezone(timezone))


def adjust_date_for_day_start(date, start_hour=6):
    """
    Adjusts a datetime object to the correct calendar day, considering that a "day"
    starts at a specific hour (e.g., 6 AM). If a time is before the start hour,
    it's considered part of the previous day.
    """
    if date.hour < start_hour:
        return date - timedelta(days=1)
    return date