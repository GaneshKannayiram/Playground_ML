import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Set page layout
st.set_page_config(page_title="Interactive Linear Regression", layout="centered")

# Toy datasets
datasets = {
    "Spending vs Salaries": pd.DataFrame({
        "X": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "Y": np.array([2, 4, 5, 4, 5, 7, 8, 7, 9, 10])
    }),
    "Hours Studied vs Marks": pd.DataFrame({
        "X": np.array([1, 3, 5, 7, 9, 11]),
        "Y": np.array([2, 6, 7, 10, 15, 20])
    })
}

# App title
st.markdown("<h1 style='text-align: center;'>📈 Interactive Linear Regression Playground</h1>", unsafe_allow_html=True)


# Dataset selection
selected_dataset = st.selectbox("🔹 Select a dataset:", list(datasets.keys()))
data = datasets[selected_dataset]

# Compute Best-Fit Line using Least Squares
X_reshaped = data["X"].values.reshape(-1, 1)
Y = data["Y"].values
lin_reg = LinearRegression().fit(X_reshaped, Y)
best_fit_m = lin_reg.coef_[0]
best_fit_b = lin_reg.intercept_

# Compute Correlation Coefficient
r_value, _ = pearsonr(data["X"], data["Y"])

# Sidebar controls
st.sidebar.header("Adjust Your Line 📏")

m = st.sidebar.slider("Slope (m)", -5.0, 5.0, 1.0, 0.1)
b = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 0.0, 0.1)

# Generate user-defined line predictions
data["Y_pred_user"] = m * data["X"] + b
errors_user = data["Y"] - data["Y_pred_user"]
sse_user = np.sum(errors_user**2)
my_mse = np.mean(errors_user**2)  # Fixed the NameError issue

# Generate best-fit line predictions
data["Y_pred_best_fit"] = best_fit_m * data["X"] + best_fit_b
errors_best_fit = data["Y"] - data["Y_pred_best_fit"]
sse_best_fit = np.sum(errors_best_fit**2)
best_fit_mse = np.mean(errors_best_fit**2)

# Sidebar controls
st.sidebar.header("Toggle Elements 🎛️")

# Checkbox for My Line
show_my_line = st.sidebar.checkbox("Show My Line", value=True)

# Dynamically update equation and MSE in two side-by-side boxes
if show_my_line:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; gap: 10px;">
            <div style="flex: 1; padding: 8px; border-radius: 5px; background-color: #333; border: 1px solid #555; color: white; font-weight: bold; text-align: center;">
                y = {m:.2f}x + {b:.2f}
            </div>
            <div style="flex: 1; padding: 8px; border-radius: 5px; background-color: #333; border: 1px solid #555; color: white; font-weight: bold; text-align: center;">
                MSE = {my_mse:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Checkbox for Best-Fit Line
show_best_fit = st.sidebar.checkbox("Show Best-Fit Line", value=True)

# Dynamically update equation and MSE in two side-by-side boxes
if show_best_fit:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; gap: 10px;">
            <div style="flex: 1; padding: 8px; border-radius: 5px; background-color: #333; border: 1px solid #555; color: white; font-weight: bold; text-align: center;">
                y = {best_fit_m:.2f}x + {best_fit_b:.2f}
            </div>
            <div style="flex: 1; padding: 8px; border-radius: 5px; background-color: #333; border: 1px solid #555; color: white; font-weight: bold; text-align: center;">
                MSE = {best_fit_mse:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Checkbox for Correlation Coefficient
show_correlation_coefficient = st.sidebar.checkbox("Show Correlation Coefficient", value=True)

if show_correlation_coefficient:
    st.sidebar.markdown(f"<div style='padding: 8px; border-radius: 5px; background-color: #333; border: 1px solid #555; color: white; font-weight: bold; text-align: center;'>r = {r_value:.2f}</div>", unsafe_allow_html=True)


# Sidebar: Residuals & Square Residuals
st.sidebar.header("Residuals & Square Residuals")

# Checkbox for Showing Residuals (My Line)
show_residuals_user = st.sidebar.checkbox("Show Residuals (My Line)", value=False)

# Checkbox for Showing Square Residuals (My Line)
show_sq_residuals_user = st.sidebar.checkbox("Show Square Residuals (My Line)", value=False)

# Checkbox for Showing Residuals (Best-Fit Line)
show_residuals_best_fit = st.sidebar.checkbox("Show Residuals (Best-Fit Line)", value=False)

# Checkbox for Showing Square Residuals (Best-Fit Line)
show_sq_residuals_best_fit = st.sidebar.checkbox("Show Square Residuals (Best-Fit Line)", value=False)

# Create the plot
fig = go.Figure()

# Scatter plot of actual data points
fig.add_trace(go.Scatter(
    x=data["X"], y=data["Y"],
    mode="markers",
    marker=dict(size=10, color="blue"),
    name="Actual Data"
))

# Best-Fit Line (Least Squares)
if show_best_fit:
    fig.add_trace(go.Scatter(
        x=data["X"], y=data["Y_pred_best_fit"],
        mode="lines",
        line=dict(color="green", width=3, dash="dot"),
        name="Best-Fit Line"
    ))

# My Line (User-Defined)
if show_my_line:
    fig.add_trace(go.Scatter(
        x=data["X"], y=data["Y_pred_user"],
        mode="lines",
        line=dict(color="red", width=3),
        name="My Line"
    ))

# Customize plot layout
fig.update_layout(
    title=f"<b>Linear Regression: y = {m:.2f}x + {b:.2f}</b>",
    xaxis_title="X",
    yaxis_title="Y",
    template="plotly_white",
    height=500
)

# Plot Residuals for My Line
if show_residuals_user:
    for i in range(len(data)):
        fig.add_trace(go.Scatter(
            x=[data["X"][i], data["X"][i]],
            y=[data["Y"][i], data["Y_pred_user"][i]],
            mode="lines",
            line=dict(color="red", dash="dot"),
            showlegend=False
        ))

# Plot Squared Residuals for My Line
if show_sq_residuals_user:
    for i in range(len(data)):
        fig.add_trace(go.Scatter(
            x=[data["X"][i], data["X"][i]],
            y=[data["Y_pred_user"][i], data["Y_pred_user"][i] + (errors_user[i]**2)],
            mode="lines",
            line=dict(color="pink", width=2),
            showlegend=False
        ))

# Plot Residuals for Best-Fit Line
if show_residuals_best_fit:
    for i in range(len(data)):
        fig.add_trace(go.Scatter(
            x=[data["X"][i], data["X"][i]],
            y=[data["Y"][i], data["Y_pred_best_fit"][i]],
            mode="lines",
            line=dict(color="green", dash="dot"),
            showlegend=False
        ))

# Plot Squared Residuals for Best-Fit Line
if show_sq_residuals_best_fit:
    for i in range(len(data)):
        fig.add_trace(go.Scatter(
            x=[data["X"][i], data["X"][i]],
            y=[data["Y_pred_best_fit"][i], data["Y_pred_best_fit"][i] + (errors_best_fit[i]**2)],
            mode="lines",
            line=dict(color="lime", width=2),
            showlegend=False
        ))


# Display the graph
st.plotly_chart(fig)

# Learning section
with st.expander("📘 Understanding the Graph"):
    st.write("""
    - **Blue points**: Actual data.  
    - **Green dashed line**: Best-Fit Line (Least Squares).  
    - **Red line**: Your manually adjusted line using sliders.  
    - **MSE**: Measures the fit of the line (lower is better!).  
    - **r (Correlation Coefficient)**: Measures the strength of the relationship.  
    """)

