import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.feature_selection import RFE


def analyze(df, plot=False):
    # Target variable is "rating_delta" and "rating" from previous day
    X = df.drop(['rating_new'], axis=1)
    y = df['rating_new']

    # get names of columns of X
    column_names = list(X.columns.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = LinearRegression()
    # model = RANSACRegressor()

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    if plot:
        # Visualize the actual vs predicted values
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Chess Rating Delta')
        plt.show()

    # Display the coefficients
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    print(coefficients)

    if plot:
        seaborn_plot(X_train)

    feature_importance(X_train, y_train)
    rfe(X_train, y_train)

    return model, column_names

def seaborn_plot(X_train):
    # Compute pairwise correlation of columns, excluding NA/null values
    correlation_matrix = X_train.corr()

    # Display the correlation matrix or use seaborn for a heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

def feature_importance(X_train, y_train):
    # Assuming X_train and y_train are already defined
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Map these importances to their corresponding variable names
    features = X_train.columns
    feature_importance_dict = dict(zip(features, importances))

    # Sort the features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    for feature, importance in sorted_features:
        print(f"{feature}: {importance}")

def rfe(X_train, y_train):
    # Initialize the RFE model and select the top 5 features
    rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5, step=1)
    rfe.fit(X_train, y_train)

    # Print the ranking of features
    ranking = rfe.ranking_
    features = X_train.columns
    ranking_dict = dict(zip(features, ranking))

    sorted_ranking = sorted(ranking_dict.items(), key=lambda x: x[1])

    for feature, rank in sorted_ranking:
        print(f"{feature}: {rank}")

def preprocess(df):
    # drop rows where date is null
    df = df.dropna(subset=['date'])

    # drop columns with names ending in "percent"
    df = df.loc[:, ~df.columns.str.endswith('percent')]

    # shift rating delta to previous day's rating
    df['rating_new'] = df['rating'] + df['rating_delta'].shift(-1)

    # drop rows where rating_new is null
    df = df.dropna(subset=['rating_new'])

    # drop rating_delta
    df = df.drop(['rating_delta'], axis=1)

    # make date index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # replace NaN values with means except in the rating_new column
    df = df.apply(lambda x: x.fillna(x.mean()) if x.name != 'rating_new' else x)

    return df

def good_baseline(df, save=False):
    high_performance_values = dict(df[df['rating_new'] > df['rating']].mean())
    del high_performance_values['rating_new']
    del high_performance_values['rating']

    low_performance_values = dict(df[df['rating_new'] < df['rating']].mean())
    del low_performance_values['rating_new']
    del low_performance_values['rating']

    # combine dicts into one, turning entries into a list
    ranges = {k: [low_performance_values[k], high_performance_values[k]] for k in low_performance_values}

    if save:
        with open('model/ranges.json', 'w') as f:
            json.dump(ranges, f)

    return ranges


def save_model(model, column_names):
    intercept = model.intercept_
    coefficients = model.coef_

    # save model
    with open('model/data.json', 'w') as f:
        json.dump({
            'intercept': intercept,
            'coefficients': list(coefficients),
            'column_names': column_names
        }, f)


if __name__ == '__main__':
    df = pd.read_csv(f"data/fitness_signals.csv")
    df = preprocess(df)
    model, column_names = analyze(df, plot=True)
    save_model(model, column_names)
    ranges = good_baseline(df, save=True)
