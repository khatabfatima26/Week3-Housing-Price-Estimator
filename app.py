import os
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from model import (
    evaluate_model,
    get_coefficients,
    load_and_prepare_data,
    train_lasso,
    train_linear_regression,
    train_ridge,
)

# Default dataset path and expected features.
DATA_PATH = Path(__file__).parent / "data" / "housing_data.csv"
TARGET_COLUMN = "Price"
FEATURE_COLUMNS = [
    "Square_Feet",
    "Bedrooms",
    "Bathrooms",
    "Age_Years",
    "Distance_City_KM",
    "Garage_Spaces",
]

st.set_page_config(
    page_title="Housing Price Estimator - Regression Model",
    page_icon="🏠",
    layout="wide",
)

# Initialize session state for consistent app behavior.
for key, default_value in {
    "data": None,
    "model": None,
    "trained_model_name": None,
    "selected_model_name": "Linear Regression",
    "alpha": 1.0,
    "metrics": {},
    "coefficients": None,
    "best_model": None,
    "data_source": "default",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def load_dataset(uploaded_file):
    """Load dataset from uploaded file or default path."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data_source = "uploaded"
            st.success("Uploaded dataset loaded successfully!")
            return df
        except Exception as exc:
            st.error(f"Failed to read uploaded file: {exc}")
            return None
    
    # Try default dataset
    try:
        df = pd.read_csv(DATA_PATH)
        st.session_state.data_source = "default"
        st.success("Default dataset loaded successfully!")
        return df
    except FileNotFoundError:
        st.info("No default dataset found. Please upload your own CSV file below.")
        return None
    except Exception as exc:
        st.error(f"Failed to load default dataset: {exc}")
        return None


def prepare_training_data(dataframe):
    """Prepare training and test data from a DataFrame."""
    if dataframe is None:
        raise ValueError("No data available for training.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Dataset must include '{TARGET_COLUMN}' column.")

    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN]

    if X.empty or y.empty:
        raise ValueError("The dataset does not contain valid feature or target data.")

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_selected_model(model_name, X_train, y_train, alpha):
    """Train the regression model chosen in the sidebar."""
    if model_name == "Ridge":
        return train_ridge(X_train, y_train, alpha=alpha)
    if model_name == "Lasso":
        return train_lasso(X_train, y_train, alpha=alpha)
    return train_linear_regression(X_train, y_train)


def compare_all_models(X_train, y_train, X_test, y_test, alpha):
    """Evaluate all candidate models to determine the best performer."""
    candidates = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Ridge": train_ridge(X_train, y_train, alpha=alpha),
        "Lasso": train_lasso(X_train, y_train, alpha=alpha),
    }
    results = {}

    for name, model in candidates.items():
        _, mse, r2 = evaluate_model(model, X_test, y_test)
        results[name] = {"mse": mse, "r2": r2}

    best_model = max(results.items(), key=lambda item: item[1]["r2"])[0]
    return results, best_model


def get_prediction_inputs(sample_data):
    """Render numeric input fields for each expected feature."""
    inputs = {}
    for feature in FEATURE_COLUMNS:
        default_value = 0.0
        if sample_data is not None and feature in sample_data:
            default_value = float(sample_data[feature].median())

        if feature in {"Bedrooms", "Bathrooms", "Garage_Spaces"}:
            inputs[feature] = st.number_input(
                f"{feature.replace('_', ' ')} 🧩",
                min_value=0,
                max_value=20,
                value=int(default_value) if default_value >= 0 else 0,
                step=1,
                key=f"input_{feature}",
            )
        else:
            max_value = 10000.0 if feature == "Square_Feet" else 500.0
            inputs[feature] = st.number_input(
                f"{feature.replace('_', ' ')} 🧩",
                min_value=0.0,
                max_value=max_value,
                value=default_value,
                step=0.1,
                key=f"input_{feature}",
            )

    return inputs


def highlight_lasso_zero(val):
    """Highlight eliminated features when Lasso shrinks coefficients to zero."""
    return "background-color: #ffefef" if val == 0 else ""


# --- Sidebar ---------------------------------------------------------------
st.sidebar.title("🏠 Housing Price Estimator")
st.sidebar.write(
    "Upload a housing dataset or use the default CSV, choose a regression model, "
    "and tune alpha for Ridge/Lasso regularization."
)

st.session_state.selected_model_name = st.sidebar.radio(
    "Model selection",
    ["Linear Regression", "Ridge", "Lasso"],
    index=0,
)

st.session_state.alpha = st.sidebar.slider(
    "Alpha for Ridge / Lasso",
    min_value=0.01,
    max_value=10.0,
    value=1.0,
    step=0.01,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload housing dataset (CSV)", type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with Streamlit for regression modeling, coefficient insights, and price prediction."
)

# Load dataset if needed.
if uploaded_file is not None:
    loaded_data = load_dataset(uploaded_file)
    if loaded_data is not None:
        st.session_state.data = loaded_data
else:
    if st.session_state.data is None:
        st.session_state.data = load_dataset(None)

# Main view header.
st.title("Housing Price Estimator - Regression Model")
st.markdown(
    "Use the tabs below to explore the data, train a regression model, "
    "inspect coefficients, and predict home prices."
)

# Create main tabs.
overview_tab, train_tab, coefficients_tab, predict_tab = st.tabs(
    ["Data Overview", "Train Model", "Coefficients", "Predict"]
)

# --- Data Overview Tab -----------------------------------------------------
with overview_tab:
    st.header("📊 Data Overview")
    if st.session_state.data is None:
        st.warning("No dataset is available. Upload a valid CSV or ensure the default dataset exists.")
    else:
        dataframe = st.session_state.data
        st.write(f"**Dataset source:** {st.session_state.data_source}")
        st.write(f"**Shape:** {dataframe.shape[0]} rows × {dataframe.shape[1]} columns")
        st.subheader("First 10 rows")
        st.dataframe(dataframe.head(10), use_container_width=True)
        st.subheader("Basic statistics")
        st.dataframe(dataframe.describe(include="all"), use_container_width=True)

# --- Train Model Tab --------------------------------------------------------
with train_tab:
    st.header("🚀 Train Model")
    if st.session_state.data is None:
        st.warning("Upload or load a dataset before training a model.")
    else:
        if st.button("Train Model"):
            try:
                st.session_state.metrics = {}
                st.session_state.coefficients = None
                st.session_state.model = None
                st.session_state.best_model = None

                with st.spinner("Training model, please wait..."):
                    progress = st.progress(0)
                    progress.progress(5)

                    X_train, X_test, y_train, y_test = prepare_training_data(st.session_state.data)
                    progress.progress(30)
                    trained_model = train_selected_model(
                        st.session_state.selected_model_name,
                        X_train,
                        y_train,
                        st.session_state.alpha,
                    )
                    progress.progress(60)
                    _, mse, r2 = evaluate_model(trained_model, X_test, y_test)
                    progress.progress(80)
                    all_metrics, best_model_name = compare_all_models(
                        X_train, y_train, X_test, y_test, st.session_state.alpha
                    )
                    progress.progress(100)

                st.session_state.model = trained_model
                st.session_state.trained_model_name = st.session_state.selected_model_name
                st.session_state.metrics = {"MSE": mse, "R2": r2}
                st.session_state.best_model = best_model_name
                try:
                    st.session_state.coefficients = get_coefficients(
                        trained_model, X_train.columns.tolist()
                    )
                except Exception as exc:
                    st.session_state.coefficients = {f: 0.0 for f in X_train.columns.tolist()}
                    st.warning(f"Could not extract coefficients: {exc}")
                st.session_state.feature_names = X_train.columns.tolist()

                st.success(f"Model trained successfully using {st.session_state.selected_model_name}.")
                st.info(f"Best model by test set R²: **{best_model_name}**")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        if st.session_state.metrics:
            metrics_row = st.columns(2)
            metrics_row[0].metric("Mean Squared Error", f"{st.session_state.metrics['MSE']:.2f}")
            metrics_row[1].metric("R² Score", f"{st.session_state.metrics['R2']:.3f}")
            st.write(f"Selected model: **{st.session_state.selected_model_name}**")
            if st.session_state.best_model:
                st.write(f"Best model across candidates: **{st.session_state.best_model}**")

# --- Coefficients Tab -------------------------------------------------------
with coefficients_tab:
    st.header("📈 Coefficients")
    if st.session_state.model is None or st.session_state.coefficients is None:
        st.warning("Train a model first to view coefficients.")
    else:
        coeff_map = st.session_state.coefficients
        feature_names = st.session_state.feature_names
        coeff_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": [coeff_map.get(feature, 0.0) for feature in feature_names],
            }
        )
        coeff_df["Impact"] = coeff_df["Coefficient"].apply(
            lambda coef: "Positive" if coef > 0 else "Negative" if coef < 0 else "Zero"
        )

        st.subheader("Feature Coefficients")
        st.bar_chart(coeff_df.set_index("Feature")["Coefficient"])

        if st.session_state.trained_model_name == "Lasso":
            st.write("Lasso may set some coefficients to zero; eliminated features are highlighted below.")
            styled = coeff_df.style.map(highlight_lasso_zero, subset=["Coefficient"])
            try:
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(coeff_df, use_container_width=True)
        else:
            st.dataframe(coeff_df, use_container_width=True)

# --- Predict Tab ------------------------------------------------------------
with predict_tab:
    st.header("🧠 Predict Price")
    if st.session_state.data is None:
        st.warning("Upload or load a dataset before using prediction.")
    elif st.session_state.model is None:
        st.warning("Train a model first before making predictions.")
    else:
        st.write(f"Predicting with model: **{st.session_state.trained_model_name}**")
        prediction_inputs = get_prediction_inputs(st.session_state.data)

        if st.button("Predict Price"):
            try:
                feature_names = st.session_state.get("feature_names", FEATURE_COLUMNS)
                pred_df = pd.DataFrame([prediction_inputs])
                for col in feature_names:
                    if col not in pred_df.columns:
                        pred_df[col] = 0.0
                pred_df = pred_df[feature_names]
                predicted_value = st.session_state.model.predict(pred_df)[0]
                st.markdown(
                    f"### 💰 Predicted Price: ${predicted_value:,.2f}"
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
