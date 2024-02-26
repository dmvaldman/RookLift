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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the logistic regression model
    model = LogisticRegression()

    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Predict on the testing set
    y_pred = model.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate the mean squared error
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f'Confusion Matrix:\n{conf_matrix}')

    feature_importance = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient']).sort_values(by='Coefficient', ascending=False)
    print(feature_importance)

    return model, scaler, column_names

def preprocess(df):
    # drop rows where date is null
    df = df.dropna(subset=['date'])

    # drop rows where rating is null
    df = df.dropna(subset=['rating'])

    # add column which is +1 if rating is higher than previous day's rating, -1 if lower
    df['rating_bool'] = df['rating_delta'].apply(lambda x: 1 if x > 0 else -1)

    # drop rating_delta
    df = df.drop(['rating_delta'], axis=1)

    # make date index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # replace NaN values with means except in the rating_delta column
    df = df.apply(lambda x: x.fillna(x.mean()) if x.name != 'rating_bool' else x)

    return df

def good_baseline(df, save=False):
    high_performance_values = dict(df[df['rating_bool'] > 0].mean())
    low_performance_values = dict(df[df['rating_bool'] < 0].mean())

    delete_columns = ['rating', 'rating_bool']
    for column in delete_columns:
        del high_performance_values[column]
        del low_performance_values[column]

    # combine dicts into one, turning entries into a list
    ranges = {k: [low_performance_values[k], high_performance_values[k]] for k in low_performance_values}

    if save:
        with open('model/ranges.json', 'w') as f:
            json.dump(ranges, f)

    return ranges


def save_model(model, scaler, column_names):
    intercept = model.intercept_[0].tolist()
    coefficients = model.coef_[0].tolist()
    scaler_mean = scaler.mean_.tolist()
    scaler_std = scaler.scale_.tolist()
    classes = model.classes_.tolist()

    # save model
    with open('model/data.json', 'w') as f:
        json.dump({
            'intercept': intercept,
            'coefficients': coefficients,
            'classes': classes,
            'column_names': column_names,
            'scaler_mean': scaler_mean,
            'scaler_std': scaler_std
        }, f)

def load_model():
    with open('model/data.json', 'r') as f:
        model_data = json.load(f)

    intercept = model_data['intercept']
    coefficients = model_data['coefficients']
    classes = model_data['classes']
    column_names = model_data['column_names']

    model = LogisticRegression()
    model.intercept_ = np.array(intercept)
    model.coef_ = np.array(coefficients)
    model.classes_ = np.array(classes)

    if 'scaler_mean' in model_data:
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
        datapoints = scaler.transform([datapoints])
    return model.predict(datapoints)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_probabilities(datapoints, model, scaler=None):
    if scaler is not None:
        datapoints = scaler.transform([datapoints])

    if isinstance(model, LogisticRegression):
        # compute manually
        z = np.dot(datapoints, model.coef_) + model.intercept_
        prob_positive_class = sigmoid(z)
        return 2 * prob_positive_class - 1
    elif isinstance(model, LinearRegression):
        # compute manually
        z = np.dot(datapoints, model.coef_) + model.intercept_
        return z
    else:
        raise ValueError("Model must be either LogisticRegression or LinearRegression")


if __name__ == '__main__':
    df = pd.read_csv(f"data/fitness_signals.csv")
    df = preprocess(df)

    # save df to csv
    df.to_csv(f"data/fitness_signals_processed.csv")

    model, scaler, column_names = analyze(df, plot=True)
    save_model(model, scaler, column_names)
    ranges = good_baseline(df, save=True)
