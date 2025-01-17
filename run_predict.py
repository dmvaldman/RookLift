import os
import dotenv
import json
import requests
import datetime
import pytz
import logging
import boto3

from download import download_range
from create_model import load_model, predict_probabilities, preprocess
from garminconnect import Garmin
from common import app, image, secrets, vol, is_local, Cron

import garth
garth.http.USER_AGENT = {"User-Agent": ("GCM-iOS-5.7.2.1")}

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('aws_access_key_id'),
    aws_secret_access_key=os.getenv('aws_secret_access_key')
)

def get_chess_rating(username):
    lichess_url = f"https://lichess.org/api/user/{username}/rating-history"
    response = requests.get(lichess_url)
    data = response.json()
    for datum in data:
        if datum['name'] == 'Blitz':
            points = datum['points']
            last_rating = points[-1][3]
            return last_rating

def get_garmin_metrics(garmin, date, features=None):
    try:
        df = download_range(garmin, date, date, save=False, force=True)
        df_processed = preprocess(df, features=features, include_rating_cols=False, num_days_lag=0, aggregate_activity=False, save=False)
    except Exception as e:
        print(f"Error: {e}")
        # use prev day if error
        date = date - datetime.timedelta(days=1)
        df = download_range(garmin, date, date, save=False, force=True)
        df_processed = preprocess(df, features=features, include_rating_cols=False, num_days_lag=0, aggregate_activity=False, save=False)
        return None

    metrics = df_processed.iloc[0].to_dict()
    return metrics

def get_datapoints(date, username, garmin, column_names, features=None):
    rating = get_chess_rating(username)
    metrics = get_garmin_metrics(garmin, date, features=features)

    values = []
    for column in column_names:
        if column == 'rating_morning':
            values.append(rating)
        else:
            values.append(metrics[column])
    return values

def compare_datapoints(datapoints, column_names, ranges, importances):
    metrics = []

    # max importance, ignoring `rating_morning` column index, and against absolute value of values
    rating_column_index = column_names.index('rating_morning')
    importance_norm = -100
    for i, importance in enumerate(importances):
        if i == rating_column_index:
            continue
        if abs(importance) > importance_norm:
            importance_norm = abs(importance)

    for feature, value, importance in zip(column_names, datapoints, importances):
        # skip 'rating'
        if feature == 'rating_morning':
            continue

        low, high = ranges[feature]

        # level between 0 and 1
        level = (value - low) / (high - low)

        # level between -1 and 1 scaled by importance, then shifted back and rescaled to [0, 1]
        level = (2 * level - 1)
        level *= abs(importance) / importance_norm
        level = (level + 1) / 2

        # replace _ with space and capitalize first letter of feature
        feature = feature.replace('_', ' ').capitalize()

        metrics.append((feature, {"importance": importance, "level": level}))

    # sort by importance, descending
    metrics.sort(key=lambda x: x[1]['importance'], reverse=True)
    return metrics

# 0 */3 * * * runs every 3 hrs
# 0 1-23/3 * * * runs every 3 hrs starting at 1am
# 0 14 * * * runs at 7am PT once a day
@app.function(
        image=image,
        secrets=[secrets],
        volumes={"/data": vol},
        schedule=Cron("0 1-23/3 * * *")
    )
def predict():
    upload = True
    features = [
        # 'active_calories',
        'activity_calories',
        # 'awake_duration',
        # 'battery_max',
        'body_battery',
        'deep_duration',
        'stress_duration',
        'light_duration',
        # 'low_stress_duration',
        'rem_duration',
        # 'sedentary_duration',
        # 'sleep_duration',
        # 'sleep_score',
        # 'sleep_stress',
        # 'steps',
        'stress_avg'
    ]

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

    try:
        datapoints = get_datapoints(date, username, garmin, column_names, features=features)
    except Exception as e:
        # use previous day because current values aren't there
        date = date - datetime.timedelta(days=1)
        datapoints = get_datapoints(date, username, garmin, column_names, features=features)

    level = predict_probabilities(datapoints, model, scaler)

    # load ranges
    with open(ranges_path, 'r') as f:
        ranges = json.load(f)

    metrics = compare_datapoints(datapoints, column_names, ranges, importances)

    print(f"Level: {level}")
    print(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    if upload:
        send_to_gist(level, metrics)

def send_to_gist(level, metrics):
    data = {
        "level": level,
        "metrics": metrics
    }

    # gist_token = os.getenv('gist_token')
    # gist_id = os.getenv('gist_id')
    # filename = "rooklift.json"

    # gist_url = f"https://api.github.com/gists/{gist_id}"

    # data_str = json.dumps(data, indent=2)
    # headers = {"Authorization": f"token {gist_token}"}
    # requests.patch(gist_url, headers=headers, json={"files": {filename: {"content": data_str}}})

    # upload file to bucket
    client.put_object(Body=json.dumps(data), Bucket='rooklift', Key='rooklift.json')

    # make file public
    client.put_object_acl(ACL='public-read', Bucket='rooklift', Key='rooklift.json')


if __name__ == '__main__':
    predict.local()