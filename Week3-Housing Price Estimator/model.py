"""Utilities for loading, training, evaluating, and saving housing regression models."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_and_prepare_data(filepath: str):
    """Load housing data from CSV and split it into train and test sets.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the housing dataset.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test as pandas objects.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset file not found: {filepath}") from exc
    except Exception as exc:
        raise ValueError(f"Unable to read dataset from {filepath}") from exc

    if "Price" not in data.columns:
        raise ValueError("The dataset must include a 'Price' column as the target variable.")

    # Separate features from the target variable.
    X = data.drop(columns=["Price"])
    y = data["Price"]

    if X.empty or y.empty:
        raise ValueError("The dataset is empty after loading.")

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    """Train a linear regression model on the provided training data."""
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as exc:
        raise RuntimeError("Failed to train linear regression model") from exc


def train_ridge(X_train, y_train, alpha: float = 1.0):
    """Train a Ridge regression model on the provided training data."""
    try:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model
    except Exception as exc:
        raise RuntimeError("Failed to train Ridge regression model") from exc


def train_lasso(X_train, y_train, alpha: float = 1.0):
    """Train a Lasso regression model on the provided training data."""
    try:
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as exc:
        raise RuntimeError("Failed to train Lasso regression model") from exc


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained regression model using test data."""
    try:
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return predictions, mse, r2
    except Exception as exc:
        raise RuntimeError("Failed to evaluate model") from exc


def get_coefficients(model, feature_names):
    """Return feature coefficients for a trained regression model.

    For Lasso regression, coefficients that shrink to zero are returned as 0.0.
    """
    if not hasattr(model, "coef_"):
        raise ValueError("The provided model does not expose coefficients.")

    if len(feature_names) != len(model.coef_):
        raise ValueError("Feature names length must match the number of model coefficients.")

    coefficients = dict(zip(feature_names, np.asarray(model.coef_, dtype=float)))
    return coefficients


def save_model(model, filename: str):
    """Save a trained model to disk using joblib."""
    try:
        joblib.dump(model, filename)
    except Exception as exc:
        raise RuntimeError(f"Failed to save model to {filename}") from exc


def load_model(filename: str):
    """Load a saved model from disk using joblib."""
    try:
        return joblib.load(filename)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model file not found: {filename}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {filename}") from exc
