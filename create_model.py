import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


def analyze(df, model_type="LogisiticRegression", num_days_lag=0):
    # Target variable is "rating_delta" and "rating" from previous day
    if num_days_lag > 0:
        df = create_lagged_features(df, num_days=num_days_lag)

    X = df.drop(['rating_bool'], axis=1)
    y = df['rating_bool'].map({-1: 0, 1: 1})

    # get names of columns of X
    column_names = list(X.columns.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    # scaler = None

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Initialize the logistic regression model
    if model_type == 'LogisticRegressionSparse':
        model = LogisticRegression(random_state=42, penalty='l1', solver='liblinear')
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=42)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_type == 'SVC':
        model = SVC(kernel='linear', random_state=42)
    else:
        raise ValueError("Model type must be either LogisticRegression, RandomForest, XGBoost, or SVC")

    print(f"Creating a {model_type} model.\n")

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate model performance
    accuracy_rf = accuracy_score(y_test, y_pred)
    precision_rf = precision_score(y_test, y_pred, zero_division=0)
    recall_rf = recall_score(y_test, y_pred)
    f1_rf = f1_score(y_test, y_pred)

    if model_type in ['LogisticRegression', 'LogisticRegressionSparse', 'SVC']:
        feature_importance = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient']).sort_values(by='Coefficient', ascending=False)
    elif model_type in ['RandomForest', 'XGBoost']:
        feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)

    print('Model performance:\n')
    print(f"Accuracy: {accuracy_rf}")
    print(f"Precision: {precision_rf}")
    print(f"Recall: {recall_rf}")
    print(f"F1: {f1_rf}")
    print(f"Confusion matrix:\n{conf_matrix}")
    print(f"Feature importance:\n{feature_importance}")

    return model, scaler, column_names

def create_lagged_features(df, num_days=2):
    lagged_df = df.copy()
    feature_columns = df.columns.difference(['rating_bool', 'date', 'rating_morning'])
    for col in feature_columns:
        for lag in range(1, num_days + 1):
            lagged_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    lagged_df = lagged_df.dropna()
    return lagged_df

def preprocess(df, save=False, save_path='data/fitness_signals_processed.csv'):
    # drop rows where date is null if it's not the index
    if 'date' in df.columns:
        df = df.dropna(subset=['date'])

    # drop rows where rating is null
    df = df.dropna(subset=['rating_morning'])

    # drop battery_max column
    df = df.drop(['battery_max'], axis=1)

    # add column `rating_bool` which is +1 if `rating_evening` column is higher than `rating_morning` column , -1 if lower
    df['rating_bool'] = (df['rating_evening'] - df['rating_morning']).shift(1).apply(lambda x: 1 if x > 0 else -1)

    # drop `rating_evening` column
    df = df.drop(['rating_evening'], axis=1)

    # process `activity_calories` column by replacing with trailing column of sum of previous 3 days, using 0 if missing
    df['activity_calories'] = df['activity_calories'].fillna(0)
    df['activity_calories'] = df['activity_calories'].rolling(window=3).sum()

    # make date index if not already
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # replace NaN values with means except in the rating_delta column
    df = df.apply(lambda x: x.fillna(x.mean()) if x.name != 'rating_bool' else x)

    if save:
        df.to_csv(save_path)

    return df

def good_baseline(df):
    df_good = df[df['rating_bool'] > 0]
    # top quantile of df values
    range_high = df_good.quantile(0.75)
    # bottom quartile of df values
    range_low = df_good.quantile(0.25)

    delete_columns = ['rating_morning', 'rating_bool']
    for column in delete_columns:
        del range_high[column]
        del range_low[column]

    # combine into one dict, looping through column names of df (note first column is the index)
    ranges = {column: [range_low[column], range_high[column]] for column in dict(range_low)}
    return ranges

def save_model(model, scaler, column_names, save_path='data/model_data.json'):
    if isinstance(model, XGBClassifier) or isinstance(model, RandomForestClassifier):
        print("Saving this type of model is not supported yet.")
        return

    intercept = model.intercept_[0].tolist()
    coefficients = model.coef_[0].tolist()
    classes = model.classes_.tolist()

    if scaler is not None:
        scaler_mean = scaler.mean_.tolist()
        scaler_std = scaler.scale_.tolist()
    else:
        scaler_mean = None
        scaler_std = None

    # save model
    with open(save_path, 'w') as f:
        json.dump({
            'type': type(model).__name__,
            'intercept': intercept,
            'coefficients': coefficients,
            'classes': classes,
            'column_names': column_names,
            'scaler_mean': scaler_mean,
            'scaler_std': scaler_std
        }, f)

def load_model(path='data/model_data.json'):
    with open(path, 'r') as f:
        model_data = json.load(f)

    type = model_data['type']
    intercept = model_data['intercept']
    coefficients = model_data['coefficients']
    classes = model_data['classes']
    column_names = model_data['column_names']

    if type == 'LogisticRegression':
        model = LogisticRegression()
    elif type == 'LogisticRegressionSparse':
        model = LogisticRegression(penalty='l1', solver='liblinear')
    elif type in ['RandomForest', 'XGBoost', 'SVC']:
        print("Loading this model not supported yet.")
        return None, None, None, None
    else:
        raise ValueError("Model type must be either LinearRegression or LogisticRegression")

    model.intercept_ = np.array(intercept)
    model.coef_ = np.array(coefficients)
    model.classes_ = np.array(classes)

    if 'scaler_mean' in model_data and model_data['scaler_mean'] is not None:
        scaler_mean = model_data['scaler_mean']
        scaler_std = model_data['scaler_std']

        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_mean)
        scaler.scale_ = np.array(scaler_std)
    else:
        scaler = None

    return model, scaler, column_names, coefficients

def predict(datapoints, model, scaler=None):
    if scaler is not None:
        datapoints = scaler.transform([datapoints])[0]
    return model.predict(datapoints)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_probabilities(datapoints, model, scaler=None):
    if scaler is not None:
        datapoints = scaler.transform([datapoints])[0]

    if isinstance(model, LogisticRegression):
        z = np.dot(datapoints, model.coef_) + model.intercept_
        return sigmoid(z)
    elif isinstance(model, LinearRegression):
        z = np.dot(datapoints, model.coef_) + model.intercept_
        return z
    else:
        raise ValueError("Model must be either LogisticRegression or LinearRegression")

def create_model(df, save=False, save_path='data/model_data.json'):
    model, scaler, column_names = analyze(df)

    if save:
        save_model(model, scaler, column_names, save_path=save_path)

    ranges = good_baseline(df, save=save)

    return model, scaler, column_names, ranges


if __name__ == '__main__':
    save = True
    num_days_lag = 0 # whether to add lagged featured to the model. currently `predict.py` doesn't support it
    model_type = 'LogisticRegressionSparse'
    # model_type = 'LogisticRegression'
    # model_type = 'RandomForest'
    # model_type = 'XGBoost'
    # model_type = 'SVC'

    df = pd.read_csv(f"data/fitness_signals.csv")
    df = preprocess(df, save=save, save_path="data/fitness_signals_processed.csv")

    print(f"Number of datapoints: {len(df)}")

    model, scaler, column_names = analyze(df, num_days_lag=num_days_lag, model_type=model_type)
    ranges = good_baseline(df)

    if save:
        save_model(model, scaler, column_names, save_path='data/model_data.json')
        with open('data/model_ranges.json', 'w') as f:
            json.dump(ranges, f)

        # test loading
        if model_type == 'LogisticRegression' or model_type == 'LogisticRegressionSparse':
            model, scaler, column_names, coefficients = load_model()
