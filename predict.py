import os
import dotenv
import json
import requests
import datetime
import numpy as np
from garminconnect import Garmin

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
        "battery_charged": body_battery_data['charged'],
        "battery_max": max_battery,
        "stress_avg": stress_data['avgStressLevel']
    }
    return metrics

def get_datapoints(date, username, garmin, column_names):
    rating = get_chess_rating(username)
    metrics = get_garmin_metrics(garmin, date)

    values = []
    for column in column_names:
        if column == 'rating':
            values.append(rating)
        else:
            values.append(metrics[column])
    return values, rating

def predict(datapoints, intercept, coefficients):
    return int(intercept + np.dot(coefficients, datapoints))

def load_model():
    with open('model/data.json', 'r') as f:
        model_data = json.load(f)

    intercept = model_data['intercept']
    coefficients = np.array(model_data['coefficients'])
    column_names = model_data['column_names']

    return intercept, coefficients, column_names

def compare_datapoints(datapoints, column_names):
    # load ranges
    with open('model/ranges.json', 'r') as f:
        ranges = json.load(f)

    for i, (feature, value) in enumerate(zip(column_names, datapoints)):
        # skip 'rating'
        if feature == 'rating':
            continue
        low, high = ranges[feature]
        if value < low:
            print(f"{feature} is below average")

if __name__ == '__main__':
    username = "dmvaldman"
    email = os.getenv('garmin_email')
    password = os.getenv('garmin_password')

    garmin = Garmin(email, password)
    garmin.login()

    today = datetime.date.today()

    intercept, coefficients, column_names = load_model()
    datapoints, rating = get_datapoints(today, username, garmin, column_names)
    predicted_rating = predict(datapoints, intercept, coefficients)

    if predicted_rating > rating:
        print(f"Your predicted rating is {predicted_rating - rating} points higher than your current rating.")
    else:
        print(f"Your predicted rating is {rating - predicted_rating} points lower than your current rating.")
        compare_datapoints(datapoints, column_names)
