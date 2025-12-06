import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# -----------------------
# Model package paths (try several)
# -----------------------
DEFAULT_MODEL_PATHS = [
    "./model_RandomForest.pkl",
    "/mnt/data/house_price_project_complete/model_RandomForest.pkl",
    "/mnt/data/model_RandomForest.pkl",
    "/mnt/data/house_price_project_pipeline_model.pkl"
]
MODEL_SAVE_PATH = DEFAULT_MODEL_PATHS[0]

# -----------------------
# Utility: Create synthetic dataset (offline)
# -----------------------
def create_synthetic_dataset(n=5000, seed=42):
    np.random.seed(seed)
    area = np.random.normal(1500, 500, size=n).clip(300, 8000)
    bedrooms = np.random.poisson(3, size=n).clip(1, 8)
    bathrooms = (np.random.poisson(2, size=n) + 0.5 * (bedrooms>2)).clip(1,6)
    age = np.random.exponential(scale=30, size=n).clip(0,120)
    lat = np.random.uniform(32.5, 42.0, size=n)
    lon = np.random.uniform(-124.5, -114.0, size=n)
    base_price = 50_000 + area * 200 + bedrooms * 10_000 + bathrooms * 7_000 - age * 300
    noise = np.random.normal(0, 40_000, size=n)
    price = (base_price + noise).clip(20_000, None)
    df = pd.DataFrame({
        "area": np.round(area, 1),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": np.round(age, 1),
        "latitude": np.round(lat, 5),
        "longitude": np.round(lon, 5),
        "price": np.round(price, 2)
    })
    return df

# -----------------------
# Train & save model package (always includes train_test_sample)
# -----------------------
def train_and_save_model(model_path=MODEL_SAVE_PATH, random_state=42, n_estimators=150):
    df = create_synthetic_dataset(n=5000, seed=random_state)
    X = df.drop(columns=["price"])
    y = df["price"].astype(float)
    numeric_feats = X.columns.tolist()
    preprocessor = ColumnTransformer([("num", StandardScaler(), numeric_feats)], remainder="drop")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_p, y_train)

    y_pred = rf.predict(X_test_p)
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    r2 = float(r2_score(y_test, y_pred))

    package = {
        "model": rf,
        "preprocessor": preprocessor,
        "numeric_features": numeric_feats,
        "metrics": {"rmse": rmse, "r2": r2},
        "train_test_sample": {"X_test": X_test.reset_index(drop=True), "y_test": y_test.reset_index(drop=True)}
    }
    os.makedirs(os.path.dirname(os.path.abspath(model_path)) or ".", exist_ok=True)
    joblib.dump(package, model_path)
    return package

# -----------------------
# Load package helper - if no train_test_sample present, create one and resave the package
# -----------------------
def load_model_package():
    for p in DEFAULT_MODEL_PATHS:
        if os.path.exists(p):
            try:
                pkg = joblib.load(p)
                # ensure train_test_sample exists
                if "train_test_sample" not in pkg or not pkg["train_test_sample"]:
                    # create a small synthetic sample for plotting (1000 rows)
                    df_sample = create_synthetic_dataset(n=1000, seed=123)
                    Xs = df_sample.drop(columns=["price"])
                    ys = df_sample["price"]
                    pkg["train_test_sample"] = {"X_test": Xs.reset_index(drop=True), "y_test": ys.reset_index(drop=True)}
                    # update metrics if missing
                    if "metrics" not in pkg:
                        try:
                            pre = pkg["preprocessor"]
                            model = pkg["model"]
                            Xp = pre.transform(Xs)
                            ypred = model.predict(Xp)
                            pkg["metrics"] = {"rmse": float(mean_squared_error(ys, ypred) ** 0.5),
                                              "r2": float(r2_score(ys, ypred))}
                        except Exception:
                            pkg["metrics"] = {}
                    # resave updated package back to same path
                    try:
                        joblib.dump(pkg, p)
                    except Exception:
                        pass
                return pkg, p
            except Exception:
                continue
    return None, None

# -----------------------
# Prediction helper
# -----------------------
def predict_from_inputs(pkg, features: dict):
    preprocessor = pkg["preprocessor"]
    model = pkg["model"]
    feature_order = pkg["numeric_features"]
    X = pd.DataFrame([features], columns=feature_order)
    X_p = preprocessor.transform(X)
    pred = model.predict(X_p)[0]
    return float(pred)

# -----------------------
# Plot helpers
# -----------------------
def plot_parity(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.4)
    mn, mx = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    ax.plot([mn, mx], [mn, mx], '--', color='red')
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Parity Plot (Actual vs Predicted)")
    return fig

def plot_residuals(y_true, y_pred):
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(y_pred, resid, alpha=0.4)
    ax.axhline(0, linestyle='--', color='red')
    ax.set_xlabel("Predicted Price")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals Plot")
    return fig

def plot_feature_importances(feature_names, importances, topk=20):
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(topk)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(fi['feature'][::-1], fi['importance'][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances (RandomForest)")
    plt.tight_layout()
    return fig, fi

def plot_price_distribution(y_series, predicted_price=None):
    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(y_series, bins=40, kde=True, ax=ax)
    if predicted_price is not None:
        ax.axvline(predicted_price, color='red', linestyle='--', linewidth=2, label='Predicted Price')
    ax.set_xlabel("Price (USD)")
    ax.set_title("Price Distribution (test/train sample)")
    if predicted_price is not None:
        ax.legend()
    return fig

def plot_feature_comparison_bar(features: dict):
    fig, ax = plt.subplots(figsize=(7,4))
    names = list(features.keys())
    vals = [features[n] for n in names]
    ax.bar(names, vals)
    ax.set_title("Input Feature Comparison")
    return fig

def plot_cv_rmse(model, X_p, y, cv=5):
    try:
        scores = cross_val_score(model, X_p, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        rmses = np.sqrt(-scores)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(range(1, cv+1), rmses, marker='o', linestyle='-')
        ax.set_xlabel("CV Fold")
        ax.set_ylabel("RMSE")
        ax.set_title("Cross-Validation RMSE per Fold")
        ax.grid(alpha=0.3)
        return fig, rmses
    except Exception as e:
        return None, None

# -----------------------
# Streamlit layout
# -----------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏡 House Price Predictor")

st.write(
    "This app will load a pre-trained model if available; otherwise it will train a RandomForest on a synthetic dataset. "
    "Evaluation and user graphs are shown interactively. The loaded/saved model package will include a 'train_test_sample' used for plotting."
)

# Load or train model package
pkg, loaded_path = load_model_package()
if pkg is None:
    st.info("No pre-trained model found — training a RandomForest model locally now (offline).")
    with st.spinner("Training model... this may take 20-60s depending on CPU"):
        pkg = train_and_save_model(model_path=MODEL_SAVE_PATH)
    st.success(f"Model trained and saved to: {MODEL_SAVE_PATH}")
else:
    st.success(f"Loaded pre-trained model from: {loaded_path}")

# show model metrics
metrics = pkg.get("metrics", {})
if metrics:
    st.subheader("Model metrics (test set from the training run)")
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{metrics.get('rmse', np.nan):.2f}")
    col2.metric("R²", f"{metrics.get('r2', np.nan):.4f}")

st.markdown("---")

# Sidebar for user inputs
st.sidebar.header("Input house features")
area = st.sidebar.number_input("Area (sqft)", min_value=200.0, max_value=20000.0, value=1500.0, step=10.0)
bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=8, value=3, step=1)
bathrooms = st.sidebar.slider("Bathrooms", min_value=1, max_value=6, value=2, step=1)
age = st.sidebar.number_input("Age (years)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=36.7, step=0.00001, format="%.5f")
longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-119.4, step=0.00001, format="%.5f")

# Button trigger
if st.sidebar.button("Predict"):
    features = {
        "area": float(area),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "age": float(age),
        "latitude": float(latitude),
        "longitude": float(longitude)
    }
    pred_price = predict_from_inputs(pkg, features)
    st.subheader("Prediction Result")
    st.metric("Predicted Price (USD)", f"${pred_price:,.2f}")

    # Determine category by comparing to training test median if present
    ref = pkg.get("train_test_sample", None)
    median_price = None
    if ref is not None:
        try:
            median_price = float(ref["y_test"].median())
        except Exception:
            median_price = None

    # category
    if median_price:
        if pred_price >= 1.25*median_price:
            category = "High"
        elif pred_price >= 0.85*median_price:
            category = "Medium"
        else:
            category = "Low"
    else:
        if pred_price >= 400000:
            category = "High"
        elif pred_price >= 200000:
            category = "Medium"
        else:
            category = "Low"

    st.write(f"**Category:** {category}")

    # Simple advice
    advice = {
        "High": "Estimated high value — check comparables and premium features.",
        "Medium": "Average value — consider minor improvements to increase price.",
        "Low": "Lower value — consider renovations or location factors."
    }
    st.info(advice[category])

    # Show user-centric graphs
    st.subheader("User-centric visuals")
    # Distribution + predicted price
    if ref is not None:
        y_test_series = ref["y_test"]
        fig_dist = plot_price_distribution(y_test_series, predicted_price=pred_price)
        st.pyplot(fig_dist)
    else:
        df_tmp = create_synthetic_dataset(n=1000)
        fig_dist = plot_price_distribution(df_tmp["price"], predicted_price=pred_price)
        st.pyplot(fig_dist)

    # Feature comparison bar
    fig_bar = plot_feature_comparison_bar(features)
    st.pyplot(fig_bar)

    st.markdown("---")

# -----------------------
# Show Evaluation & Training Graphs below (always visible)
# -----------------------
st.subheader("Model Evaluation & Training Performance")

ref = pkg.get("train_test_sample", None)
if ref is not None:
    X_test_df = ref["X_test"]
    y_test_series = ref["y_test"]
    preprocessor = pkg["preprocessor"]
    model = pkg["model"]
    X_test_p = preprocessor.transform(X_test_df)
    y_pred = model.predict(X_test_p)

    # parity
    fig_parity = plot_parity(y_test_series, y_pred)
    st.pyplot(fig_parity)

    # residuals
    fig_resid = plot_residuals(y_test_series, y_pred)
    st.pyplot(fig_resid)

    # feature importances (if RF)
model = pkg["model"]  # this is your trained RandomForest model

if hasattr(model, "feature_importances_"):
    feat_names = pkg["numeric_features"]         # keep your correct feature order
    importances = model.feature_importances_

    # Build sorted DataFrame
    fi = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Insert rank column 1...N
    fi.insert(0, "rank", range(1, len(fi) + 1))

    # Save tidy CSV
    fi.to_csv("feature_importances.csv", index=False)

    # Plot
    plt.figure(figsize=(8,6))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    st.pyplot(plt)

    # Show clean table
    st.subheader("Feature Importance Table")
    st.dataframe(fi)


    # Cross-val RMSE (training performance)
    st.markdown("**Cross-validation RMSE (5-fold)**")
    try:
        sample_idx = np.random.choice(range(X_test_p.shape[0]), size=min(500, X_test_p.shape[0]), replace=False)
        X_cv = X_test_p[sample_idx]
        y_cv = y_test_series.values[sample_idx]
        fig_cv, rmses = plot_cv_rmse(pkg["model"], X_cv, y_cv, cv=5)
        if fig_cv is not None:
            st.pyplot(fig_cv)
            st.write(f"Per-fold RMSE: {np.round(rmses,2).tolist()}  — mean = {np.round(rmses.mean(),2)}")
    except Exception as e:
        st.write("Could not compute CV RMSE due to: ", e)
else:
    st.write("No test sample available inside the model package to produce evaluation plots.")

st.markdown("---")
st.write("© 2025 House Price Predictor App")

