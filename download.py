import requests
import datetime
import os
import dotenv
import time
import json
from garminconnect import Garmin
import pandas as pd

dotenv.load_dotenv()


# ratings are beginning of day ratings
def get_lichess_ratings(username, start_date, end_date, game_type='Blitz', save=False):
    save_path = f"data/daily_ratings.json"
    if os.path.exists(save_path):
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
            json.dump(daily_ratings, f)

    return daily_ratings

def get_daily_stress(garmin, start_date, end_date, save=False):
    save_path = f"data/daily_stress.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    # loop through days between start_date and end_date
    daily_stress = []
    for day in range((end_date - start_date).days):
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
            json.dump(daily_stress, f)

    return daily_stress

def get_body_battery(garmin, start_date, end_date, save=False):
    save_path = f"data/daily_battery.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    # if start_date is more than 30 days behind end_date, batch into 30-day chunks and combine
    daily_battery = []
    if (end_date - start_date).days > 30:
        for day in range((end_date - start_date).days // 30 + 1):
            start = start_date + datetime.timedelta(days=30 * day)
            end = min(start_date + datetime.timedelta(days=30 * (day + 1)), end_date)
            body_battery = garmin.get_body_battery(start.isoformat(), end.isoformat())
            for data in body_battery:
                if data['charged'] is None and data['drained'] is None:
                    continue

                battery_values = [datum[1] for datum in data['bodyBatteryValuesArray']]
                if battery_values[0] is None:
                    max_battery = None
                else:
                    max_battery = max(battery_values)

                # convert to negative for drained
                day_battery = {
                    "date": data['date'],
                    "battery_charged": data['charged'],
                    "battery_max": max_battery
                }
                daily_battery.append(day_battery)
            time.sleep(0.1)
    else:
        body_battery = garmin.get_body_battery(start_date.isoformat(), end_date.isoformat())

    # sort by date in calendar order
    daily_battery.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_battery, f)

    return daily_battery

def get_sleep_score(garmin, start_date, end_date, save=False):
    save_path = f"data/daily_sleep.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    # loop through days between startdate and today
    daily_sleep_scores = []
    for day in range((end_date - start_date).days):
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
        }

        daily_sleep_scores.append(data)
        time.sleep(0.1)

    # sort by date in calendar order
    daily_sleep_scores.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_sleep_scores, f)

    return daily_sleep_scores

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

def main(lichess_username, garmin, save=False):
    today = datetime.date.today()

    # find earliest date with data for both lichess and garmin
    # game_type is out of ['Blitz', 'Bullet', 'Rapid', 'Classical'] with the most volume
    # TODO: add support for multiple game types
    date_lichess, game_type = find_earliest_record_date_and_game_type(lichess_username)
    date_garmin = find_earliest_record_date(garmin, start=date_lichess, end=today)

    # start_date is max of dates from lichess and garmin
    start_date = date_garmin
    end_date = datetime.date.today()

    daily_ratings = get_lichess_ratings(lichess_username, start_date, end_date, game_type=game_type, save=save)
    daily_battery = get_body_battery(garmin, start_date, end_date, save=save)
    daily_stress = get_daily_stress(garmin, start_date, end_date, save=save)
    daily_sleep = get_sleep_score(garmin, start_date, end_date, save=save)

    signals = [daily_ratings, daily_battery, daily_stress, daily_sleep]

    df = signals_to_df(signals, save=save)
    return df

def signals_to_df(signals, save=False):
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

    if save:
        # save to data dir as csv
        df_pivoted.to_csv(f"data/fitness_signals.csv")

    return df_pivoted


if __name__ == "__main__":
    lichess_username = "dmvaldman"
    save = True

    email = os.getenv('garmin_email')
    password = os.getenv('garmin_password')

    garmin = Garmin(email, password)
    garmin.login()

    signals_df = main(lichess_username, garmin, save=save)