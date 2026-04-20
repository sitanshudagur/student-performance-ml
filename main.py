import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
def load_data():
    return pd.read_csv('data/student_data.csv')


# Prepare data
def prepare_data(data):
    X = data[['StudyHours', 'SleepHours', 'Attendance']]
    y = data['Marks']
    return X, y


# Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return predictions, mae, r2


def main():
    print("Loading data...")
    data = load_data()

    print("Preparing data...")
    X, y = prepare_data(data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    predictions, mae, r2 = evaluate_model(model, X_test, y_test)

    print("\n--- Results ---")
    print("Predicted Marks:", predictions)
    print("MAE:", mae)
    print("R2 Score:", r2)

    print("\nModel Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef}")

    print("Intercept:", model.intercept_)


if __name__ == "__main__":
    main()