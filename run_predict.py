import os
import dotenv
import json
import requests
import datetime
import pytz
import logging

from download import download_range
from create_model import load_model, predict_probabilities, preprocess
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

def get_garmin_metrics(garmin, date, features=None):
    df = download_range(garmin, date, date, save=False, force=True)
    df_processed = preprocess(df, features=features, include_rating_cols=False, num_days_lag=0, aggregate_activity=False, save=False)
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
    for feature, value, importance in zip(column_names, datapoints, importances):
        # skip 'rating'
        if feature == 'rating_morning':
            continue

        # Remove this line when we're confident about negative importance
        importance = abs(importance)

        low, high = ranges[feature]

        # level between 0 and 1
        level = (value - low) / (high - low)

        # level between -.5 and .5 scaled by importance, then shifted back
        level = importance * (level - 0.5) + 0.5

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
def predict(upload=False, features=None, ):
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
    datapoints = get_datapoints(date, username, garmin, column_names, features=features)
    level = predict_probabilities(datapoints, model, scaler)

    # load ranges
    with open(ranges_path, 'r') as f:
        ranges = json.load(f)

    metrics = compare_datapoints(datapoints, column_names, ranges, importances)

    print(f"Level: {level}")
    print(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    if upload:
        send_to_jsonbin(level, metrics)

def send_to_jsonbin(level, metrics):
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
    print('JSBIN PUT response:\n', json.dumps(response_data, indent=2))

if __name__ == '__main__':
    upload = True
    predict.local(upload=upload)