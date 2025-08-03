import os
import dotenv
import json
import datetime
import pytz
import logging
import boto3
import modal
import numpy as np
import pandas as pd

from modal_defs import image, secrets, vol, app, Cron, is_local
from create_model import load_model, predict_probabilities, preprocess
from garmin_client import GarminClient
from lichess_client import LichessClient

import garth
garth.http.USER_AGENT = {"User-Agent": ("GCM-iOS-5.7.2.1")}

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

# https://rooklift.s3.us-west-1.amazonaws.com/rooklift.json
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('aws_access_key_id'),
    aws_secret_access_key=os.getenv('aws_secret_access_key')
)

def get_garmin_metrics(garmin_client, date, features=None):
    """
    Fetches Garmin metrics for a specific date, processes them, and returns a dictionary.
    Retries with the previous day on failure.
    """
    metrics_dict = garmin_client.download_day(date)

    # Convert the dictionary of metrics into a DataFrame for preprocessing
    df = pd.DataFrame([metrics_dict])
    df_processed = preprocess(df, features=features, include_rating_cols=False, num_days_lag=0, aggregate_activity=False, save=False)
    return df_processed.iloc[0].to_dict()


def get_datapoints(date, lichess_client, garmin_client, column_names, features=None):
    rating = lichess_client.download_current(game_type='Blitz')
    metrics = get_garmin_metrics(garmin_client, date, features=features)

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

def run_prediction(save=True):
    features = [
        'active_calories',
        # 'activity_calories',
        # 'awake_duration',
        # 'battery_max',
        # 'body_battery',
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

    # Determine if running locally or on Modal
    data_dir = "data" if is_local else "/data"
    model_path = os.path.join(data_dir, 'model_data.json')
    ranges_path = os.path.join(data_dir, 'model_ranges.json')

    if not is_local:
        vol.resolve()

    lichess_username = os.getenv("lichess_username")
    garmin_email = os.getenv('garmin_email')
    garmin_password = os.getenv('garmin_password')

    lichess_client = LichessClient(lichess_username)
    garmin_client = GarminClient(garmin_email, garmin_password)

    # get today's date in the current timezone
    current_timezone = pytz.timezone('US/Pacific')
    date = datetime.datetime.now(current_timezone).date()

    model, scaler, column_names, importances = load_model(model_path)

    datapoints = get_datapoints(date, lichess_client, garmin_client, column_names, features=features)
    level = predict_probabilities(datapoints, model, scaler)

    # Whether data is fresh for the day or some fields have yet to update
    fresh = not np.isnan(datapoints).any()

    # load ranges
    with open(ranges_path, 'r') as f:
        ranges = json.load(f)

    metrics = compare_datapoints(datapoints, column_names, ranges, importances)

    print(f"Level: {level}")
    print(f"Metrics:\n{json.dumps(metrics, indent=2)}")
    print(f"Fresh:\n{fresh}")

    data = {
        "level": level,
        "metrics": metrics,
        "fresh": fresh
    }

    if save:
        save_to_S3(data)

    return data

def save_to_S3(data):
    # upload file to bucket
    s3_client.put_object(Body=json.dumps(data), Bucket='rooklift', Key='rooklift.json')

    # make file public
    s3_client.put_object_acl(ACL='public-read', Bucket='rooklift', Key='rooklift.json')


@app.function(
        image=image,
        secrets=[secrets],
        volumes={"/data": vol}
    )
@modal.fastapi_endpoint()
def predict_webhook():
    """Webhook endpoint for on-demand predictions"""
    result = run_prediction(save=True)
    return result

@app.function(
        image=image,
        secrets=[secrets],
        volumes={"/data": vol},
        schedule=Cron("0 9-21/4 * * *")
    )
def predict_scheduled():
    """Scheduled function that runs at 8am, 9am, and every 4 hours after 9am"""
    run_prediction(save=True)

if __name__ == '__main__':
    run_prediction(save=False)