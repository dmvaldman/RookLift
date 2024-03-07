import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def analyze(df, model_type="LogisiticRegression", plot=False):
    # Target variable is "rating_delta" and "rating" from previous day
    X = df.drop(['rating_bool'], axis=1)
    y = df['rating_bool']

    column_names = X.columns

    scaler = StandardScaler()
    # scaler = None

    if scaler is not None:
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Cross-validation scores
    evaluate_model(model, X, y, cv=5, plot=plot)

    # Calculate model performance
    accuracy_rf = accuracy_score(y_test, y_pred)
    precision_rf = precision_score(y_test, y_pred, zero_division=0)
    recall_rf = recall_score(y_test, y_pred)
    f1_rf = f1_score(y_test, y_pred)

    if model_type in ['LogisticRegression', 'LogisticRegressionSparse', 'SVC']:
        feature_importance = pd.DataFrame(model.coef_[0], index=column_names, columns=['Coefficient']).sort_values(by='Coefficient', ascending=False)
    elif model_type in ['RandomForest', 'XGBoost']:
        feature_importance = pd.DataFrame(model.feature_importances_, index=column_names, columns=['Importance']).sort_values(by='Importance', ascending=False)

    print('Model performance:\n')
    print(f"Accuracy: {accuracy_rf}")
    print(f"Precision: {precision_rf}")
    print(f"Recall: {recall_rf}")
    print(f"F1: {f1_rf}")
    print(f"Confusion matrix:\n{conf_matrix}")
    print(f"Feature importance:\n{feature_importance}")

    return model, scaler, column_names.to_list()

# Function to perform cross-validation and plot learning curves
def evaluate_model(model, X, y, cv=5, plot=False):
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Average CV Score: {cv_scores.mean()}")

    if plot:
        # Learning curves
        train_sizes, train_scores, validation_scores = learning_curve(model, X, y, cv=cv, n_jobs=-1,
                                                                    train_sizes=np.linspace(0.1, 1.0, 10))

        # Calculate mean and standard deviation for training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate mean and standard deviation for validation set scores
        validation_mean = np.mean(validation_scores, axis=1)
        validation_std = np.std(validation_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Training score", color="blue", marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.15)

        plt.plot(train_sizes, validation_mean, label="Cross-validation score", color="green", marker='o')
        plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color="green", alpha=0.15)

        plt.title("Learning Curve")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy Score")
        plt.legend(loc="best")
        plt.grid()
        plt.show()

def create_lagged_features(df, num_days=2):
    lagged_df = df.copy()
    feature_columns = df.columns.difference(['rating_bool', 'date', 'rating_morning'])
    for col in feature_columns:
        for lag in range(1, num_days + 1):
            lagged_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    lagged_df = lagged_df.dropna()
    return lagged_df

def preprocess(df, features=None, aggregate_activity=False, include_rating_cols=True, num_days_lag=0, save=False, save_path='data/fitness_signals_processed.csv'):
    # drop all rows except the classic rows
    if features is not None:
        if include_rating_cols:
            features += ['rating_morning', 'rating_evening']
        print('Avail Columns:', df.columns)
        df = df[features]

    # drop rows where date is null if it's not the index
    if 'date' in df.columns:
        df = df.dropna(subset=['date'])

    # rename `body_battery_during_sleep` to `body_battery` if it exists
    if 'body_battery_during_sleep' in df.columns:
        df.rename(columns={'body_battery_during_sleep': 'body_battery'}, inplace=True)

    # rename `high_stress_duration` to `stress_duration` if it exists
    if 'stress_duration' in df.columns:
        df.rename(columns={'stress_duration': 'high_stress_duration'}, inplace=True)

    # not called when running `predict.py`
    if include_rating_cols:
        # drop rows where rating is null
        df = df.dropna(subset=['rating_morning'])
        # add column `rating_bool` which is +1 if `rating_evening` column is higher than `rating_morning` column , -1 if lower
        df['rating_bool'] = (df['rating_evening'] - df['rating_morning']).shift(1).apply(lambda x: 1 if x > 0 else -1)
        # drop `rating_evening` column
        df = df.drop(['rating_evening'], axis=1)

    # process `activity_calories` column by replacing with trailing column of sum of previous 2 days, using 0 if missing
    if 'activity_calories' in df.columns:
        df['activity_calories'] = df['activity_calories'].fillna(0)
        if aggregate_activity: df['activity_calories'] = df['activity_calories'].rolling(window=2).sum()

    # # process active_kilocalories column by replacing with trailing column of sum of previous 2 days, using 0 if missing
    if 'active_kilocalories' in df.columns:
        df['active_kilocalories'] = df['active_kilocalories'].fillna(0)
        if aggregate_activity: df['active_kilocalories'] = df['active_kilocalories'].rolling(window=2).sum()

    # # process `active_seconds` column by replacing with trailing column of sum of previous 2 days, using 0 if missing
    if 'active_seconds' in df.columns:
        df['active_seconds'] = df['active_seconds'].fillna(0)
        if aggregate_activity: df['active_seconds'] = df['active_seconds'].rolling(window=2).sum()

    # make date index if not already
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # replace NaN values with means except in the rating_bool column
    df = df.apply(lambda x: x.fillna(x.mean()) if x.name != 'rating_bool' else x)

    # add lagged features
    if num_days_lag > 0:
        df = create_lagged_features(df, num_days=num_days_lag)

    if save:
        df.to_csv(save_path)

    return df

def good_baseline(df):
    df_good = df[df['rating_bool'] > 0]
    # top quantile of df values
    range_high = df_good.quantile(0.80)
    # bottom quartile of df values
    range_low = df_good.quantile(0.20)

    # hack for activity calories which are usually 0. take non-zero and do quartile of those
    if 'activity_calories' in df.columns:
        range_high['activity_calories'] = df_good['activity_calories'][df_good['activity_calories'] > 0].quantile(0.70)
        range_low['activity_calories'] = 0

    # lots of missing values so use tighter bounds
    if 'body_battery' in df.columns:
        non_zeros_indices = df_good['body_battery'] > 0
        range_high['body_battery'] = df_good['body_battery'][non_zeros_indices].quantile(0.90)
        range_low['body_battery'] = df_good['body_battery'][non_zeros_indices].quantile(0.10)

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

    # If LogisticRegressionSparse, remove coefficients that are 0 from coefficients and column_names and intercept
    # if model_type == 'LogisticRegressionSparse':
    #     thresh = 1e-6
    #     column_names = [column_names[i] for i, c in enumerate(coefficients) if abs(c) > thresh]
    #     coefficients = [coefficients for i, c in enumerate(coefficients) if abs(c) > thresh]

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

if __name__ == '__main__':
    save = True
    num_days_lag = 0 # whether to add lagged featured to the model. currently `predict.py` doesn't support it
    aggregate_activity = False # whether to aggregate activity data by summing previous N days
    plot = False

    # For historical reasons
    # classic features: ['activity_calories', 'awake_duration', 'deep_duration', 'light_duration', 'rem_duration', 'sleep_duration', 'sleep_score', 'sleep_stress', 'stress_avg']

    features = [
        # 'active_calories',
        'activity_calories',
        # 'awake_duration',
        # 'battery_max',
        'body_battery_during_sleep',
        'deep_duration',
        'high_stress_duration',
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

    # model_type = 'LogisticRegressionSparse'
    model_type = 'LogisticRegression'
    # model_type = 'SVC'
    # model_type = 'RandomForest'
    # model_type = 'XGBoost'

    df = pd.read_csv(f"data/fitness_signals.csv")
    df = preprocess(
        df,
        features=features,
        num_days_lag=num_days_lag,
        aggregate_activity=aggregate_activity,
        save=save,
        save_path="data/fitness_signals_processed.csv")

    print(f"Number of datapoints: {len(df)}")

    model, scaler, column_names = analyze(df, model_type=model_type, plot=plot)
    ranges = good_baseline(df)

    if save:
        save_model(model, scaler, column_names, save_path='data/model_data.json')
        with open('data/model_ranges.json', 'w') as f:
            json.dump(ranges, f, indent=2)

        # test loading
        if model_type == 'LogisticRegression' or model_type == 'LogisticRegressionSparse':
            model, scaler, column_names, coefficients = load_model()
