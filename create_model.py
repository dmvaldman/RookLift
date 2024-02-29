import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def analyze(df, plot=False):
    # Target variable is "rating_delta" and "rating" from previous day
    X = df.drop(['rating_bool'], axis=1)
    y = df['rating_bool']

    # get names of columns of X
    column_names = list(X.columns.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    # scaler = None

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Initialize the logistic regression model
    model = LogisticRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate the mean squared error
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f'Confusion Matrix:\n{conf_matrix}')

    feature_importance = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient']).sort_values(by='Coefficient', ascending=False)
    print(feature_importance)

    return model, scaler, column_names

def preprocess(df, save=False, save_path='data/fitness_signals_processed.csv'):
    # drop rows where date is null if it's not the index
    if 'date' in df.columns:
        df = df.dropna(subset=['date'])

    # drop rows where rating is null
    df = df.dropna(subset=['rating_morning'])

    # drop battery_max column
    df = df.drop(['battery_max'], axis=1)

    # add column `rating_bool` which is +1 if `rating_evening` column is higher than `rating_morning` column , -1 if lower
    df['rating_bool'] = (df['rating_evening'] - df['rating_morning']).apply(lambda x: 1 if x >= 0 else -1)

    # drop `rating_evening` column
    df = df.drop(['rating_evening'], axis=1)

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
    high_performance_values = dict(df[df['rating_bool'] > 0].mean())
    low_performance_values = dict(df[df['rating_bool'] < 0].mean())

    delete_columns = ['rating_morning', 'rating_bool']
    for column in delete_columns:
        del high_performance_values[column]
        del low_performance_values[column]

    # combine dicts into one, turning entries into a list
    ranges = {k: [low_performance_values[k], high_performance_values[k]] for k in low_performance_values}

    return ranges


def save_model(model, scaler, column_names, save_path='data/model_data.json'):
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

    if type == 'LinearRegression':
        model = LinearRegression()
    elif type == 'LogisticRegression':
        model = LogisticRegression()
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

    return model, scaler, column_names

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
    model, scaler, column_names = analyze(df, plot=True)

    if save:
        save_model(model, scaler, column_names, save_path=save_path)

    ranges = good_baseline(df, save=save)

    return model, scaler, column_names, ranges


if __name__ == '__main__':
    save = True
    df = pd.read_csv(f"data/fitness_signals.csv")
    df = preprocess(df, save=save, save_path="data/fitness_signals_processed.csv")

    model, scaler, column_names = analyze(df, plot=True)
    ranges = good_baseline(df)

    if save:
        save_model(model, scaler, column_names, save_path='data/modal_data.json')
        with open('data/model_ranges.json', 'w') as f:
            json.dump(ranges, f)
