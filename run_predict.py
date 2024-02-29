import os
import dotenv
import json
import requests
import datetime
import pytz

from create_model import load_model, predict_probabilities
from garminconnect import Garmin
from common import stub, image, secrets, vol, is_local, Cron

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

    battery_values = [datum[1] for datum in body_battery_data['bodyBatteryValuesArray']]
    max_battery = max(battery_values)

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

def compare_datapoints(datapoints, column_names, ranges):
    above_average_metrics = []
    below_average_metrics = []
    for i, (feature, value) in enumerate(zip(column_names, datapoints)):
        # skip 'rating'
        if feature == 'rating_morning':
            continue
        low, high = ranges[feature]
        if value < low:
            below_average_metrics.append(feature)
        elif value > high:
            above_average_metrics.append(feature)

    str = "Metrics above average:\n" + "\n".join(above_average_metrics) + "\n\n" + "Metrics below average:\n" + "\n".join(below_average_metrics)
    return str

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

    model, scaler, column_names = load_model(model_path)
    datapoints = get_datapoints(date, username, garmin, column_names)
    level = predict_probabilities(datapoints, model, scaler)

    # load ranges
    with open(ranges_path, 'r') as f:
        ranges = json.load(f)

    compare_str = compare_datapoints(datapoints, column_names, ranges)
    level_str = "You should take a break" if level < .5 else "You should play chess!"

    send_to_jsbin(level, level_str, compare_str)

def send_to_jsbin(level, level_str, compare_str):
    X_ACCESS_KEY = os.getenv('JSONBIN_ACCESS_KEY')
    X_MASTER_KEY = os.getenv('JSONBIN_MASTER_KEY')

    url = "https://api.jsonbin.io/v3/b/65cc1fd01f5677401f2ef548"

    headers = {
        "X-Master-Key": X_MASTER_KEY,
        "X-Access-Key": X_ACCESS_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "level": level,
        "advice": level_str,
        "compare": compare_str
    }

    response = requests.put(url, headers=headers, json=data)
    response_data = response.json()
    print('JSBIN PUT response: ', response_data)

if __name__ == '__main__':
    predict.local()