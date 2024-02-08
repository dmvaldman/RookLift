import requests
import datetime
import os
import dotenv
import time
import json
from garminconnect import Garmin

dotenv.load_dotenv()

email = os.getenv('garmin_email')
password = os.getenv('garmin_password')

def heuristic_to_numeric(heuristic):
    if heuristic == "EXCELLENT":
        return 3
    if heuristic == "GOOD":
        return 2
    if heuristic == "FAIR":
        return 1
    if heuristic == "POOR":
        return 0

def get_lichess_ratings(username, start_date, end_date, game_type='Blitz', save=False):
    save_path = f"data/daily_ratings_{start_date}-{end_date}.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    url = f"https://lichess.org/api/user/{username}/rating-history"
    response = requests.get(url)
    data = response.json()
    ratings_data = []
    for datum in data:
        if datum['name'] == game_type:
            points = datum['points']
            for rating_data in points:
                year, month, day, rating = rating_data
                date = str(datetime.date(year, month + 1, day))
                data_formatted = {
                    "date": date,
                    "rating": rating
                }
                ratings_data.append(data_formatted)

    # sort by date in calendar order
    ratings_data.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    # convert array from [{date, value1, value2}, ...] to {date: [dates], value1: [value1s], value2: [value2s]}
    daily_ratings = {}
    for key in ratings_data[0].keys():
        daily_ratings[key] = [x[key] for x in ratings_data]

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_ratings, f)

    return daily_ratings

def get_daily_stress(garmin, start_date, end_date, save=False):
    save_path = f"data/daily_stress_{start_date}-{end_date}.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    stress_data = []
    # loop through days between start_date and end_date
    for day in range((end_date - start_date).days):
        date = start_date + datetime.timedelta(days=day)
        data = garmin.get_stress_data(date.isoformat())
        data_formatted = {
            "date": data['calendarDate'],
            "stress_max": data['maxStressLevel'],
            "stress_avg": data['avgStressLevel'],
        }
        stress_data.append(data_formatted)
        time.sleep(0.2)

    # sort by date in calendar order
    stress_data.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    # convert array from [{date, value1, value2}, ...] to {date: [dates], value1: [value1s], value2: [value2s]}
    daily_stress = {}
    for key in stress_data[0].keys():
        daily_stress[key] = [x[key] for x in stress_data]

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_stress, f)

    return daily_stress

def get_body_battery(garmin, start_date, end_date, save=False):
    save_path = f"data/daily_battery_{start_date}-{end_date}.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            battery_data = json.load(f)
        return battery_data

    # if start_date is more than 30 days behind end_date, batch into 30-day chunks and combine
    battery_data = []
    if (end_date - start_date).days > 30:
        for day in range((end_date - start_date).days // 30 + 1):
            start = start_date + datetime.timedelta(days=30 * day)
            end = start_date + datetime.timedelta(days=30 * (day + 1))
            body_battery = garmin.get_body_battery(start.isoformat(), end.isoformat())
            for data in body_battery:
                if data['charged'] is None and data['drained'] is None:
                    continue
                day_battery = {
                    "date": data['date'],
                    "charged": data['charged'],
                    "drained": data['drained'],
                    "delta": data['charged'] - data['drained']
                }
                battery_data.append(day_battery)
            time.sleep(0.1)
    else:
        body_battery = garmin.get_body_battery(start_date.isoformat(), end_date.isoformat())

    # sort by date in calendar order
    battery_data.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    # convert array from [{date, value1, value2}, ...] to {date: [dates], value1: [value1s], value2: [value2s]}
    daily_battery = {}
    for key in battery_data[0].keys():
        daily_battery[key] = [x[key] for x in battery_data]

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_battery, f)

    return daily_battery

def get_sleep_score(garmin, start_date, end_date, save=False):
    save_path = f"data/daily_sleep_{start_date}-{end_date}.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            daily_battery = json.load(f)
        return daily_battery

    sleep_scores = []
    # loop through days between startdate and today
    for day in range((end_date - start_date).days):
        day = start_date + datetime.timedelta(days=day)
        data = garmin.get_sleep_data(day.isoformat())

        if 'sleepScores' not in data['dailySleepDTO']:
            continue

        data = {
            "date": data['dailySleepDTO']['calendarDate'],
            "duration_score": heuristic_to_numeric(data['dailySleepDTO']['sleepScores']['totalDuration']['qualifierKey']),
            "stress_score": heuristic_to_numeric(data['dailySleepDTO']['sleepScores']['stress']['qualifierKey']),
            "awake_count_score": heuristic_to_numeric(data['dailySleepDTO']['sleepScores']['awakeCount']['qualifierKey']),
            "rem_percentage_score": heuristic_to_numeric(data['dailySleepDTO']['sleepScores']['remPercentage']['qualifierKey']),
            "light_percentage_score": heuristic_to_numeric(data['dailySleepDTO']['sleepScores']['lightPercentage']['qualifierKey']),
            "deep_percentage_score": heuristic_to_numeric(data['dailySleepDTO']['sleepScores']['deepPercentage']['qualifierKey']),
            "restless_score": heuristic_to_numeric(data['dailySleepDTO']['sleepScores']['restlessness']['qualifierKey']),
            "light_calue": data['dailySleepDTO']['sleepScores']['lightPercentage']['value'],
            "rem_value": data['dailySleepDTO']['sleepScores']['remPercentage']['value'],
            "deep_value": data['dailySleepDTO']['sleepScores']['deepPercentage']['value'],
            "sleep_stress": data['dailySleepDTO']['avgSleepStress'],
            "sleep_duration": data['dailySleepDTO']['sleepTimeSeconds'],
            "lightSleepSeconds": data['dailySleepDTO']['lightSleepSeconds'],
            "remSleepSeconds": data['dailySleepDTO']['remSleepSeconds'],
            "deepSleepSeconds": data['dailySleepDTO']['deepSleepSeconds'],
        }

        sleep_scores.append(data)
        time.sleep(0.1)

    # sort by date in calendar order
    sleep_scores.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    # convert array from [{date, value1, value2}, ...] to {date: [dates], value1: [value1s], value2: [value2s]}
    daily_sleep_scores = {}
    for key in sleep_scores[0].keys():
        daily_sleep_scores[key] = [x[key] for x in sleep_scores]

    if save:
        # save to data dir as json
        with open(save_path, "w") as f:
            json.dump(daily_sleep_scores, f)

    return daily_sleep_scores

if __name__ == "__main__":
    lichess_username = "dmvaldman"
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)  # Select past year
    save = True

    daily_ratings = get_lichess_ratings(lichess_username, start_date, end_date, save=True)

    garmin = Garmin(email, password)
    garmin.login()

    daily_battery = get_body_battery(garmin, start_date, end_date, save=True)
    daily_stress = get_daily_stress(garmin, start_date, end_date, save=True)
    daily_sleep = get_sleep_score(garmin, start_date, end_date, save=True)