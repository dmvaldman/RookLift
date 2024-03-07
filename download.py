import requests
import datetime
import os
import dotenv
import time
import json
import pandas as pd
import lichess.api
import pytz

from datetime import timedelta
from collections import defaultdict
from garminconnect import Garmin

from common import stub, image, secrets, is_local

dotenv.load_dotenv()

def datetime_to_timestamp(dt, timezone="US/Pacific"):
    # if dt is instance of Date, convert to Datetime at {start_hour}am
    if isinstance(dt, datetime.date):
        dt = datetime.datetime.combine(dt, datetime.time(0, 0, 1))

    # Ensure dt is timezone-aware. Convert it to the desired timezone without replacing tzinfo directly.
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = pytz.utc.localize(dt)  # Localize as UTC if naive

    current_timezone = pytz.timezone(timezone)
    datetime_local = dt.astimezone(current_timezone)
    timestamp = datetime_local.timestamp()
    timestamp_ms = int(timestamp * 1000)
    return timestamp_ms

def timestamp_to_datetime(timestamp_ms, timezone="US/Pacific"):
    # Convert milliseconds to seconds for utcfromtimestamp and ensure it's in the correct timezone
    timestamp_seconds = timestamp_ms / 1000.0
    timestamp_datetime = datetime.datetime.utcfromtimestamp(timestamp_seconds)
    current_timezone = pytz.timezone(timezone)
    timestamp_local = timestamp_datetime.replace(tzinfo=pytz.utc).astimezone(current_timezone)
    return timestamp_local

def adjust_date_for_day_start(date, start_hour=6):
    """Adjusts the datetime to the correct day considering the day starts at a specific hour."""
    if date.hour < start_hour:
        # Consider it as the previous day
        return date - timedelta(days=1)
    return date

def get_lichess_ratings(username, start_date, end_date, game_type='Blitz', save=False, save_dir="data", force=False):
    save_path = os.path.join(save_dir, "daily_ratings.json")
    if not force and os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_ratings = json.load(f)
        return daily_ratings

    perfType = game_type.lower()
    timezone = "US/Pacific"
    start_hour = 6 # 6am local time

    # convert start_date to epoch
    since = datetime_to_timestamp(start_date, timezone=timezone)
    until = datetime_to_timestamp(end_date, timezone=timezone)
    game_generator = lichess.api.user_games(username, perfType=perfType, rated=True, since=since, until=until)

    # loop through games and mark the user's rating in the beginning and end of each day
    # beginning of day is at start_hour
    ratings = []
    for game in game_generator:
        timestamp_ms = game['createdAt']
        date = timestamp_to_datetime(timestamp_ms, timezone=timezone)

        if game['players']['white']['user']['name'] == username:
            rating_before = game['players']['white']['rating']
            rating_after = rating_before + game['players']['white']['ratingDiff']
        else:
            rating_before = game['players']['black']['rating']
            rating_after = rating_before + game['players']['black']['ratingDiff']

        data_formatted = {
            "date": date,
            "rating_before": rating_before,
            "rating_after": rating_after
        }
        ratings.append(data_formatted)

    # sort by date
    ratings.sort(key=lambda x: x['date'])

    # Group by the adjusted date
    daily_ratings_dict = defaultdict(lambda: {"rating_before": None, "rating_after": None})
    for record in ratings:
        adjusted_date = adjust_date_for_day_start(record["date"], start_hour=start_hour).date()  # Ignoring time part after adjustment
        if daily_ratings_dict[adjusted_date]["rating_before"] is None or record["date"].time() < daily_ratings_dict[adjusted_date]["date"].time():
            daily_ratings_dict[adjusted_date]["rating_before"] = record["rating_before"]
            daily_ratings_dict[adjusted_date]["date"] = record["date"]  # Storing the datetime for comparison
        if daily_ratings_dict[adjusted_date]["rating_after"] is None or record["date"].time() >= daily_ratings_dict[adjusted_date]["date"].time():
            daily_ratings_dict[adjusted_date]["rating_after"] = record["rating_after"]
            daily_ratings_dict[adjusted_date]["date"] = record["date"]  # Storing the datetime for comparison

    # Convert the defaultdict to a list of dictionaries
    daily_ratings = [{"date": str(date), "rating_morning": data["rating_before"], "rating_evening": data["rating_after"]} for date, data in daily_ratings_dict.items()]

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_ratings, f, indent=2)

    return daily_ratings

# ratings are beginning of day ratings
def get_lichess_ratings_old(username, start_date, end_date, game_type='Blitz', save=False, save_dir="data", force=False):
    save_path = os.path.join(save_dir, "daily_ratings.json")
    if not force and os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    url = f"https://lichess.org/api/user/{username}/rating-history"
    response = requests.get(url)
    data = response.json()
    daily_ratings = []
    prev_rating = None
    for datum in data:
        if datum['name'] == game_type:
            points = datum['points']
            for rating_data in points:
                year, month, day, rating = rating_data
                date = datetime.date(year, month + 1, day)

                rating_delta = rating - prev_rating if prev_rating is not None else 0

                # limit date to being between start_date and end_date
                if start_date <= date <= end_date:
                    data_formatted = {
                        "date": str(date),
                        "rating": rating,
                        "rating_delta": rating_delta,
                    }
                    daily_ratings.append(data_formatted)
                prev_rating = rating

    # sort by date in calendar order
    daily_ratings.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_ratings, f, indent=2)

    return daily_ratings

def get_daily_stress(garmin, start_date, end_date, save=False, save_dir="data", force=False):
    save_path = os.path.join(save_dir, "daily_stress.json")
    if not force and os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    # loop through days between start_date and end_date
    daily_stress = []
    for day in range((end_date - start_date).days + 1):
        date = start_date + datetime.timedelta(days=day)
        data = garmin.get_stress_data(date.isoformat())
        data_formatted = {
            "date": data['calendarDate'],
            "stress_avg": data['avgStressLevel']
        }
        daily_stress.append(data_formatted)
        time.sleep(0.2)

    # sort by date in calendar order
    daily_stress.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_stress, f, indent=2)

    return daily_stress

def get_body_battery_during_sleep(garmin, start_date, end_date, save=False, save_dir="data", force=False):
    save_path = os.path.join(save_dir, "daily_battery_sleep.json")
    if not force and os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery_sleep = json.load(f)
        return daily_battery_sleep

    # loop through days between start_date and end_date
    body_battery_sleep = []
    for day in range((end_date - start_date).days + 1):
        date = start_date + datetime.timedelta(days=day)
        data = garmin.get_user_summary(date.isoformat())

        # shift date back one day
        date_shifted = datetime.datetime.strptime(data['calendarDate'], '%Y-%m-%d').date() - datetime.timedelta(days=0)

        data_formatted = {
            "date": str(date_shifted),
            "body_battery_during_sleep": data['bodyBatteryDuringSleep']
        }

        body_battery_sleep.append(data_formatted)
        time.sleep(0.2)

    # sort by date in calendar order
    body_battery_sleep.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(body_battery_sleep, f, indent=2)

    return body_battery_sleep

def get_daily_summary(garmin, start_date, end_date, save=False, save_dir="data", force=False):
    save_path = os.path.join(save_dir, "daily_summary.json")
    if not force and os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    # loop through days between start_date and end_date
    daily_summary = []
    for day in range((end_date - start_date).days + 1):
        date = start_date + datetime.timedelta(days=day)
        data = garmin.get_user_summary(date.isoformat())

        calendarDate = data['calendarDate']
        # shift up calendary date one day
        date_shifted = datetime.datetime.strptime(calendarDate, '%Y-%m-%d').date() + datetime.timedelta(days=1)

        data_formatted = {
            "date": str(date_shifted),
            "steps": data['totalSteps'],
            # "highly_active_seconds": data['highlyActiveSeconds'],
            # "active_seconds": data['activeSeconds'],
            "sedentary_duration": data['sedentarySeconds'],
            "high_stress_duration": data['highStressDuration'],
            "low_stress_duration": data['lowStressDuration'],
            "active_calories": data['activeKilocalories'],
            # "bodyBatteryHighestValue": data['bodyBatteryHighestValue'],
        }
        daily_summary.append(data_formatted)
        time.sleep(0.2)

    # sort by date in calendar order
    daily_summary.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_summary, f, indent=2)

    return daily_summary

def get_body_battery(garmin, start_date, end_date, save=False, save_dir="data", force=False):
    save_path = os.path.join(save_dir, "daily_battery.json")
    if not force and os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    def get_body_battery_chunk(start, end):
        body_battery = garmin.get_body_battery(start.isoformat(), end.isoformat())
        values = []
        for data in body_battery:
            if data['charged'] is None and data['drained'] is None:
                continue

            battery_values = [datum[1] for datum in data['bodyBatteryValuesArray'] if datum[1] is not None]
            max_battery = max(battery_values) if battery_values else None

            # convert to negative for drained
            day_battery = {
                "date": data['date'],
                # "battery_charged": data['charged'],
                "battery_max": max_battery
            }
            values.append(day_battery)
        return values

    # if start_date is more than 30 days behind end_date, batch into 30-day chunks and combine
    daily_battery = []
    if (end_date - start_date).days > 30:
        for day in range((end_date - start_date).days // 30 + 1):
            start = start_date + datetime.timedelta(days=30 * day)
            end = min(start_date + datetime.timedelta(days=30 * (day + 1)), end_date)
            daily_battery.extend(get_body_battery_chunk(start, end))
            time.sleep(0.1)
    else:
        daily_battery = get_body_battery_chunk(start_date, end_date)

    # sort by date in calendar order
    daily_battery.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_battery, f, indent=2)

    return daily_battery

def get_sleep_score(garmin, start_date, end_date, save=False, save_dir="data", force=False):
    save_path = os.path.join(save_dir, "daily_sleep.json")
    if not force and os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    # loop through days between startdate and today
    daily_sleep_scores = []
    for day in range((end_date - start_date).days + 1):
        day = start_date + datetime.timedelta(days=day)
        data = garmin.get_sleep_data(day.isoformat())

        if 'sleepScores' not in data['dailySleepDTO']:
            continue

        data = {
            "date": data['dailySleepDTO']['calendarDate'],
            "sleep_stress": data['dailySleepDTO']['avgSleepStress'],
            "light_duration": data['dailySleepDTO']['lightSleepSeconds'],
            "rem_duration": data['dailySleepDTO']['remSleepSeconds'],
            "deep_duration": data['dailySleepDTO']['deepSleepSeconds'],
            "sleep_duration": data['dailySleepDTO']['sleepTimeSeconds'],
            "sleep_score": data['dailySleepDTO']['sleepScores']['overall']['value'],
            "awake_duration": data['dailySleepDTO']['awakeSleepSeconds']
        }

        daily_sleep_scores.append(data)
        time.sleep(0.1)

    # sort by date in calendar order
    daily_sleep_scores.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_sleep_scores, f, indent=2)

    return daily_sleep_scores

def get_activities(garmin, start_date, end_date, save=False, save_dir="data", force=False):
    save_path = os.path.join(save_dir, "daily_activities.json")
    if not force and os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_activies = json.load(f)
        return daily_activies

    def get_activities_chunk(start, end):
        values = []
        activities = garmin.get_activities_by_date(start.isoformat(), end.isoformat())

        for data in activities:
            start_time = data['startTimeLocal'] #YYYY-MM-DD HH:MM:SS
            date = start_time.split(' ')[0]
            # use for next day's results
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date() + datetime.timedelta(days=1)
            # convert to negative for drained

            # search if date already exists in values
            date_exists = False
            for day in values:
                if day['date'] == str(date):
                    date_exists = True
                    day['activity_calories'] += data['calories']
                    break

            if not date_exists:
                day_activities = {
                    "date": str(date),
                    "activity_calories": data['calories']
                }
                values.append(day_activities)

        return values

    daily_activities = []
    if (end_date - start_date).days > 30:
        for day in range((end_date - start_date).days // 30 + 1):
            start = start_date + datetime.timedelta(days=30 * day)
            end = min(start_date + datetime.timedelta(days=30 * (day + 1)), end_date)
            daily_activities.extend(get_activities_chunk(start, end))
            time.sleep(0.1)
    else:
        daily_activities = get_activities_chunk(start_date, end_date)

    # sort by date in calendar order
    daily_activities.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_activities, f, indent=2)

    # sort by date in calendar order
    daily_activities.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))
    return daily_activities

def find_earliest_record_date_and_game_type(lichess_username):
    url = f"https://lichess.org/api/user/{lichess_username}/rating-history"
    response = requests.get(url)
    data = response.json()
    dates_and_volume = {}
    for datum in data:
        earliest_date = None
        points = datum['points']
        if len(points) == 0 or datum['name'] not in ['Bullet', 'Blitz', 'Rapid', 'Classical']:
            continue
        for rating_data in points:
            year, month, day, rating = rating_data
            date = datetime.date(year, month + 1, day)
            if earliest_date is None or date < earliest_date:
                earliest_date = date
        volume = len(points)
        dates_and_volume[datum['name']] = (earliest_date, volume)

    # return earliest date and game type game type with most volume
    max_games = 0
    max_games_type = None
    for game_type, (date, volume) in dates_and_volume.items():
        if volume > max_games:
            max_games = volume
            max_games_type = game_type

    return dates_and_volume[max_games_type][0], max_games_type

def find_earliest_record_date(garmin, start, end):
    # recursive binary search for earliest date with data
    if (end - start).days <= 1:
        return end

    mid_date = start + (end - start) // 2

    data = garmin.get_body_battery(mid_date.isoformat())
    if data[0]['charged'] is not None:
        return find_earliest_record_date(garmin, start=start, end=mid_date)
    else:
        return find_earliest_record_date(garmin, start=mid_date, end=end)

def signals_to_df(signals):
    # Flatten the list of lists into a single list
    flattened_list = [item for sublist in signals for item in sublist]

    # Create a DataFrame from the flattened list
    df = pd.DataFrame(flattened_list)

    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Since we have different keys for each metric, we need to melt the DataFrame first
    # then pivot it to get the desired shape.
    df_melted = df.melt(id_vars=['date'], var_name='metric', value_name='value')

    # Pivot table to reshape the data
    df_pivoted = pd.pivot_table(df_melted, values='value', index=['date'], columns=['metric'], aggfunc='first')

    return df_pivoted

def download(lichess_username, garmin, save=False, save_path="data/fitness_signals.csv", save_dir="data", force=False):
    today = datetime.date.today()

    # find earliest date with data for both lichess and garmin
    # game_type is out of ['Blitz', 'Bullet', 'Rapid', 'Classical'] with the most volume
    # TODO: add support for multiple game types
    date_lichess, game_type = find_earliest_record_date_and_game_type(lichess_username)
    date_garmin = find_earliest_record_date(garmin, start=date_lichess, end=today)

    # start_date is max of dates from lichess and garmin
    start_date = date_garmin
    end_date = today

    df = download_range(garmin, start_date, end_date, lichess_username=lichess_username, game_type=game_type, save=save, save_dir=save_dir, force=force)

    if save:
        df.to_csv(save_path)

    return df

def download_range(garmin, start_date, end_date, lichess_username=None, game_type=None, save=False, save_dir="data", force=False):
    signals = []

    signals.append(get_body_battery(garmin, start_date, end_date, save=save, save_dir=save_dir, force=force))
    signals.append(get_daily_stress(garmin, start_date, end_date, save=save, save_dir=save_dir, force=force))
    signals.append(get_sleep_score(garmin, start_date, end_date, save=save, save_dir=save_dir, force=force))
    signals.append(get_activities(garmin, start_date, end_date, save=save, save_dir=save_dir, force=force))
    signals.append(get_daily_summary(garmin, start_date, end_date, save=save, save_dir=save_dir, force=force))
    signals.append(get_body_battery_during_sleep(garmin, start_date, end_date, save=save, save_dir=save_dir, force=force))

    if lichess_username:
        daily_ratings = get_lichess_ratings(lichess_username, start_date, end_date, game_type=game_type, save=save, save_dir=save_dir, force=force)
        signals.append(daily_ratings)

    df = signals_to_df(signals)

    return df

if __name__ == '__main__':
    save = True
    force = True

    save_dir = "data" if is_local else "/data"
    save_path = os.path.join(save_dir, "fitness_signals.csv")

    lichess_username = os.getenv("lichess_username")
    garmin_email = os.getenv('garmin_email')
    garmin_password = os.getenv('garmin_password')

    garmin = Garmin(garmin_email, garmin_password)
    garmin.login()

    signals_df = download(lichess_username, garmin, save=save, save_dir=save_dir, save_path=save_path, force=force)