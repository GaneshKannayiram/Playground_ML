import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure page
st.set_page_config(
    page_title="SVR Playground", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎛️ Interactive Support Vector Regression Playground")

# Generate synthetic datasets with dynamic controls
def generate_data(dataset_type, n_samples=100, noise=0.1, outlier_count=5, outlier_strength=20):
    if dataset_type == "Linear":
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=42)
    elif dataset_type == "Nonlinear (Sine)":
        X = np.linspace(0, 10, n_samples).reshape(-1, 1)
        y = np.sin(X).ravel() + noise * np.random.randn(n_samples)
    elif dataset_type == "With Outliers":
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=42)
        outlier_indices = np.random.choice(n_samples, size=outlier_count, replace=False)
        y[outlier_indices] += outlier_strength
    return X, y

# Sidebar controls
st.sidebar.header("⚙️ Controls")

# Dataset Configuration
dataset_type = st.sidebar.selectbox(
    "Dataset Type",
    ["Linear", "Nonlinear (Sine)", "With Outliers"],
    index=0,
    help="Choose data structure to analyze SVR behavior"
)

noise = st.sidebar.slider(
    "Noise Level", 
    0.1, 1.0, 0.3 if dataset_type == "Nonlinear (Sine)" else 1.0, 0.1,
    help="Amount of random noise in data"
)

if dataset_type == "With Outliers":
    outlier_count = st.sidebar.slider("Outlier Count", 1, 10, 5)
    outlier_strength = st.sidebar.slider("Outlier Strength", 5.0, 30.0, 20.0)

# SVR Parameters
st.sidebar.header("SVR Hyperparameters")

kernel_info = {
    "linear": "Linear kernel (works best when data is linearly separable)",
    "rbf": "Radial Basis Function (handles non-linear patterns)",
    "poly": "Polynomial kernel (captures polynomial relationships)"
}

kernel = st.sidebar.selectbox(
    "Kernel Type",
    options=list(kernel_info.keys()),
    format_func=lambda x: f"{x} - {kernel_info[x].split('(')[0]}",
    index=1,
    help="Select the kernel function for SVR"
)

C = st.sidebar.slider(
    "Regularization (C)",
    0.01, 100.0, 1.0, 0.01,
    help="Low C: Wider margin, more errors allowed. High C: Fit training data closely (risk overfitting)."
)

epsilon = st.sidebar.slider(
    "Epsilon (ε-tube width)",
    0.01, 1.0, 0.1, 0.01,
    help="Width of error-insensitive zone. Larger ε = more points ignored."
)

# Dynamic gamma help text
gamma_disabled = kernel not in ['rbf', 'poly']
gamma_help = ("Controls RBF curvature" if kernel == "rbf" else 
             "Polynomial feature scale" if kernel == "poly" else 
             "Not used in linear kernel")
gamma = st.sidebar.slider(
    "Gamma (Kernel Coefficient)",
    0.001, 10.0, 0.1, 0.001,
    disabled=gamma_disabled,
    help=gamma_help
) if not gamma_disabled else "auto"

degree = st.sidebar.slider(
    "Degree (Poly kernel)",
    2, 5, 3,
    disabled=(kernel != "poly"),
    help="Degree of polynomial kernel"
)

# Generate data (without caching for real-time updates)
X, y = generate_data(
    dataset_type, 
    noise=noise,
    outlier_count=outlier_count if dataset_type == "With Outliers" else 0,
    outlier_strength=outlier_strength if dataset_type == "With Outliers" else 0
)

# Train SVR model (optimized initialization)
model = SVR(
    C=C, 
    kernel=kernel, 
    gamma=gamma if kernel in ['rbf', 'poly'] else 'scale', 
    epsilon=epsilon, 
    degree=degree if kernel == 'poly' else 3  # Explicitly unused for non-poly
)
model.fit(X, y)
y_pred = model.predict(X)

# Create prediction line with edge case handling
# Added ±1 padding to handle near-constant X values
X_range_padding = max(1, 0.1 * (X.max() - X.min()))  # Adaptive padding
X_test = np.linspace(X.min() - X_range_padding, 
                     X.max() + X_range_padding, 
                     300).reshape(-1, 1)
y_test = model.predict(X_test)
upper_bound = y_test + epsilon
lower_bound = y_test - epsilon

# Identify points outside ε-tube
outside_mask = (y > y_pred + epsilon) | (y < y_pred - epsilon)

# Main Plot
fig = go.Figure()

# Data points (color by ε-tube membership)
fig.add_trace(go.Scatter(
    x=X[~outside_mask].squeeze(), 
    y=y[~outside_mask],
    mode='markers',
    name='Inside ε-tube',
    marker=dict(color='green', size=8),
    hoverinfo='x+y',
    visible=True
))

# Outside ε-tube points (now more visible)
fig.add_trace(go.Scatter(
    x=X[outside_mask].squeeze(), 
    y=y[outside_mask],
    mode='markers',
    name='Outside ε-tube',
    marker=dict(
        color='firebrick',  # Changed to high-contrast red
        size=10,
        symbol='x',
        line=dict(width=1, color='white')
    ),
    hoverinfo='x+y',
    visible=True
))

# Prediction line and ε-tube
fig.add_trace(go.Scatter(
    x=X_test.squeeze(), 
    y=y_test,
    mode='lines',
    name='SVR Prediction',
    line=dict(color='red', width=3),
    visible=True
))

fig.add_trace(go.Scatter(
    x=np.concatenate([X_test.squeeze(), X_test.squeeze()[::-1]]),
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255,0,0,0.1)',
    line=dict(color='rgba(255,0,0,0.5)', width=1),
    name='ε-tube (soft margin)',
    visible=True
))

# Support vectors toggle
show_svs = st.sidebar.checkbox("Show Support Vectors", value=True)
fig.add_trace(go.Scatter(
    x=X[model.support_].squeeze(),
    y=y[model.support_],
    mode='markers',
    name='Support Vectors',
    marker=dict(
        color='lime', 
        size=12, 
        line=dict(color='black', width=2)
    ),
    hovertext=[f"SV {i+1}: ({x:.2f}, {y:.2f})" 
              for i, (x, y) in enumerate(zip(X[model.support_].squeeze(), 
                                           y[model.support_]))],
    visible=show_svs
))

fig.update_layout(
    title=f"SVR Visualization (Kernel: {kernel}, C={C}, ε={epsilon})",
    xaxis_title="Input Feature",
    yaxis_title="Target Value",
    yaxis_range=[min(y)-1, max(y)+1],
    hovermode='closest',
    height=600,
    dragmode='zoom'
)

st.plotly_chart(fig, use_container_width=True)

# Residual Plot with enhanced tooltips
residuals = y - y_pred
fig_residuals = go.Figure()
fig_residuals.add_trace(go.Scatter(
    x=X.squeeze(), 
    y=residuals,
    mode='markers',
    marker=dict(
        color=['red' if abs(r) > epsilon else 'blue' for r in residuals],
        size=8
    ),
    name='Residuals',
    hovertext=[f"X: {x:.2f}<br>Actual: {act:.2f}<br>Predicted: {pred:.2f}<br>Error: {err:.2f}" 
              for x, act, pred, err in zip(X.squeeze(), y, y_pred, residuals)],
    visible=True
))
fig_residuals.add_hline(y=0, line_dash="dash")
fig_residuals.add_hrect(
    y0=-epsilon, y1=epsilon, 
    fillcolor="rgba(0,100,80,0.1)", 
    line_width=0,
    annotation_text="ε-tube", 
    annotation_position="top left"
)
fig_residuals.update_layout(
    title="Residual Analysis",
    xaxis_title="Input Feature",
    yaxis_title="Prediction Error",
    showlegend=False
)
st.plotly_chart(fig_residuals, use_container_width=True)

# Metrics and Data Stats
st.subheader("Performance Metrics")
col1, col2, col3 = st.columns(3)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

col1.metric("MSE", f"{mse:.2f}", 
           delta="Low error" if mse < 0.5 else "High error", 
           delta_color="inverse")
col2.metric("R² Score", f"{r2:.2f}", 
           delta="Good fit" if r2 > 0.7 else "Poor fit", 
           delta_color="normal" if r2 > 0.7 else "inverse")
col3.metric("MAE", f"{mae:.2f}")

# Dataset Statistics with improved formatting
with st.expander("📊 Dataset Statistics", expanded=False):
    stats_df = pd.DataFrame({
        'Feature': ['X', 'y'],
        'Mean': [float(np.mean(X)), float(np.mean(y))],
        'Std Dev': [float(np.std(X)), float(np.std(y))],
        'Min': [float(np.min(X)), float(np.min(y))],
        'Max': [float(np.max(X)), float(np.max(y))]
    })
    st.dataframe(stats_df.style.format("{:.2f}", subset=['Mean', 'Std Dev', 'Min', 'Max']), 
                hide_index=True)

# Model Details with enhanced explanations
with st.expander("🔍 Model Details", expanded=False):
    st.write(f"**Number of Support Vectors:** {len(model.support_)}")
    st.caption("Support vectors are data points that either: "
              "(1) lie outside the ε-tube, or "
              "(2) are exactly on the ε-tube boundary.")
    
    sv_coords = pd.DataFrame({
        'X Value': X[model.support_].squeeze(),
        'Actual Y': y[model.support_],
        'Predicted Y': y_pred[model.support_],
        'Error': residuals[model.support_]
    })
    st.write("**Support Vector Coordinates:**")
    st.dataframe(sv_coords.style.format("{:.2f}"), height=200)

# Mobile responsiveness
st.markdown("""
<style>
@media (max-width: 600px) {
    .sidebar .sidebar-content {
        width: 100px;
    }
    .stButton>button {
        min-width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)