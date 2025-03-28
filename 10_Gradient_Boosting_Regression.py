import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree

# Set page layout
st.set_page_config(page_title="Interactive Gradient Boosting Regression Playground", layout="wide")

# Load dataset
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Linear Regression":
        X, y = make_regression(
            n_samples=100, n_features=1, noise=20, random_state=42
        )
        X = pd.DataFrame(X, columns=["Feature 1"])
    elif dataset_name == "Nonlinear Regression":
        X = np.linspace(-10, 10, 100).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.normal(0, 10, size=100)
        X = pd.DataFrame(X, columns=["Feature 1"])
    elif dataset_name == "Multiple Features":
        X, y = make_regression(
            n_samples=100, n_features=2, noise=20, random_state=42
        )
        X = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    return X, y

# App title
st.markdown("<h1 style='text-align: center;'>📈 Interactive Gradient Boosting Regression Playground</h1>", unsafe_allow_html=True)

# Dataset selection
st.sidebar.header("Dataset Selection")
dataset_name = st.sidebar.selectbox("🔹 Select a dataset:", ["Linear Regression", "Nonlinear Regression", "Multiple Features"])
X, y = load_data(dataset_name)

# Sidebar controls
st.sidebar.header("Adjust Hyperparameters 📏")
n_estimators = st.sidebar.slider("Number of Estimators (n_estimators)", 1, 200, 100, 1)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3, 1)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, 1)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1, 1)

# Train Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42,
)
model.fit(X, y)

# Generate predictions
X_test = np.linspace(X.min().min(), X.max().max(), 100).reshape(-1, 1)
if X.shape[1] == 1:
    y_pred = model.predict(X_test)
else:
    # For multiple features, use the mean of Feature 2 for visualization
    X_test = np.column_stack([X_test, np.full_like(X_test, X["Feature 2"].mean())])
    y_pred = model.predict(X_test)

# Plot actual data points and regression line (for 1D datasets)
if X.shape[1] == 1:
    fig = px.scatter(X, x="Feature 1", y=y, title="Gradient Boosting Regression")
    fig.add_trace(px.line(x=X_test.flatten(), y=y_pred, color_discrete_sequence=["red"]).data[0])
    fig.update_traces(marker=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("**Note:** Regression line visualization is only available for 1D datasets.")

# Model Metrics
st.header("Model Metrics")
y_train_pred = model.predict(X)
mse = mean_squared_error(y, y_train_pred)
mae = mean_absolute_error(y, y_train_pred)
r2 = r2_score(y, y_train_pred)

st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Residuals Plot
st.header("Residuals Plot")
residuals = y - y_train_pred
fig_residuals = px.scatter(x=y_train_pred, y=residuals, title="Residuals vs Predicted Values")
fig_residuals.add_trace(px.line(x=[y_train_pred.min(), y_train_pred.max()], y=[0, 0], color_discrete_sequence=["red"]).data[0])
st.plotly_chart(fig_residuals, use_container_width=True)

# Comparison with Default Model
st.header("Comparison with Default Model")
default_model = GradientBoostingRegressor(random_state=42)
default_model.fit(X, y)
default_y_pred = default_model.predict(X)
default_mse = mean_squared_error(y, default_y_pred)
default_mae = mean_absolute_error(y, default_y_pred)
default_r2 = r2_score(y, default_y_pred)

st.write(f"**Default Model MSE:** {default_mse:.2f}")
st.write(f"**Default Model MAE:** {default_mae:.2f}")
st.write(f"**Default Model R² Score:** {default_r2:.2f}")

# Feature Importance
st.header("Feature Importance")
feature_importance = model.feature_importances_
fig_importance = px.bar(x=X.columns, y=feature_importance, title="Feature Importance")
st.plotly_chart(fig_importance, use_container_width=True)

# Decision Tree Visualization
st.header("Decision Tree Visualization")
st.write("Visualization of the first tree in the Gradient Boosting ensemble:")
fig_tree, ax = plt.subplots(figsize=(12, 8))
plot_tree(model.estimators_[0, 0], filled=True, feature_names=X.columns, ax=ax)
st.pyplot(fig_tree)