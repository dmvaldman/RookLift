import os
import dotenv
import json
import requests
import datetime
import pytz
import logging

from create_model import load_model, predict_probabilities
from garminconnect import Garmin
from common import stub, image, secrets, vol, is_local, Cron

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

def get_chess_rating(username):
    lichess_url = f"https://lichess.org/api/user/{username}/rating-history"
    response = requests.get(lichess_url)
    data = response.json()
    for datum in data:
        if datum['name'] == 'Blitz':
            points = datum['points']
            last_rating = points[-1][3]
            return last_rating

def get_garmin_metrics(garmin, date):
    body_battery_data = garmin.get_body_battery(date.isoformat())[0]
    stress_data = garmin.get_stress_data(date.isoformat())
    sleep_data = garmin.get_sleep_data(date.isoformat())

    battery_values = [datum[1] for datum in body_battery_data['bodyBatteryValuesArray'] if datum[1] is not None]
    max_battery = max(battery_values) if battery_values else None

    # logging.debug(f"Dictionary: {json.dumps(sleep_data['dailySleepDTO'], indent=2)}")
    print(f"Dictionary: {json.dumps(sleep_data['dailySleepDTO'], indent=2)}")

    metrics = {
        "sleep_stress": sleep_data['dailySleepDTO']['avgSleepStress'],
        "light_duration": sleep_data['dailySleepDTO']['lightSleepSeconds'],
        "rem_duration": sleep_data['dailySleepDTO']['remSleepSeconds'],
        "deep_duration": sleep_data['dailySleepDTO']['deepSleepSeconds'],
        "sleep_duration": sleep_data['dailySleepDTO']['sleepTimeSeconds'],
        "sleep_score": sleep_data['dailySleepDTO']['sleepScores']['overall']['value'],
        "battery_max": max_battery,
        "stress_avg": stress_data['avgStressLevel']
    }
    return metrics

def get_datapoints(date, username, garmin, column_names):
    rating = get_chess_rating(username)
    metrics = get_garmin_metrics(garmin, date)

    values = []
    for column in column_names:
        if column == 'rating_morning':
            values.append(rating)
        else:
            values.append(metrics[column])
    return values

def compare_datapoints(datapoints, column_names, ranges, importances):
    metrics = []
    for feature, value, importance in zip(column_names, datapoints, importances):
        # skip 'rating'
        if feature == 'rating_morning':
            continue

        low, high = ranges[feature]

        level = (value - low) / (high - low)
        if importance < 0:
            level *= -1

        # replace _ with space and capitalize first letter of feature
        feature = feature.replace('_', ' ').capitalize()

        metrics.append((feature, {"importance": importance, "level": level}))

    # sort by importance, descending
    metrics.sort(key=lambda x: abs(x[1]['importance']), reverse=True)
    return metrics

# 0 */2 * * * runs every 2 hrs
# 0 14 * * * runs at 7am PT once a day
@stub.function(
        image=image,
        secrets=[secrets],
        volumes={"/data": vol},
        schedule=Cron("0 */3 * * *")
    )
def predict():
    data_dir = 'data' if is_local else '/data'
    model_path = os.path.join(data_dir, 'model_data.json')
    ranges_path = os.path.join(data_dir, 'model_ranges.json')

    if not is_local:
        vol.resolve()

    username = os.getenv("lichess_username")
    email = os.getenv('garmin_email')
    password = os.getenv('garmin_password')

    garmin = Garmin(email, password)
    garmin.login()

    # get today's date in the current timezone
    current_timezone = pytz.timezone('US/Pacific')
    date = datetime.datetime.now(current_timezone).date()

    model, scaler, column_names, importances = load_model(model_path)
    datapoints = get_datapoints(date, username, garmin, column_names)
    level = predict_probabilities(datapoints, model, scaler)

    # load ranges
    with open(ranges_path, 'r') as f:
        ranges = json.load(f)

    metrics = compare_datapoints(datapoints, column_names, ranges, importances)

    send_to_jsbin(level, metrics)

def send_to_jsbin(level, metrics):
    X_ACCESS_KEY = os.getenv('JSONBIN_ACCESS_KEY')
    X_MASTER_KEY = os.getenv('JSONBIN_MASTER_KEY')

    url = os.getenv('JSONBIN_URL')

    headers = {
        "X-Master-Key": X_MASTER_KEY,
        "X-Access-Key": X_ACCESS_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "level": level,
        "metrics": metrics
    }

    response = requests.put(url, headers=headers, json=data)
    response_data = response.json()
    print('JSBIN PUT response: ', response_data)

if __name__ == '__main__':
    predict.local()