import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page layout
st.set_page_config(page_title="Interactive Polynomial Regression", layout="centered")

# Predefined Ice Cream Selling Data (updated with CSV data)
ice_cream_data = {
    "Temperature": [-4.662262677220208, -4.316559446725467, -4.213984764590729, -3.9496610890515707, -3.578553716228682, -3.455711698065576, -3.1084401208909964, -3.0813033243034563, -2.672460827006454, -2.652286792936049, -2.6514980333001315, -2.288263998488389, -2.111869690297304, -1.8189376094349368, -1.6603477296372017, -1.3263789834948425, -1.1731232680778254, -0.773330043103446, -0.6737528018380356, -0.14963486653359837, -0.03615649768268739, -0.033895285571446534, 0.00860769873161413, 0.14924457404675835, 0.6887809076106148, 0.6935988725293257, 0.8749050291584157, 1.0241808138155706, 1.2407116187783346, 1.3598126741393184, 1.7400000122653544, 1.850551925836895, 1.9993103690372853, 2.075100596577272, 2.318591239633373, 2.4719459973351453, 2.784836463321575, 2.8317602113138346, 2.959932091478899, 3.020874314267157, 3.211366144342004, 3.270044068238683, 3.316072519311853, 3.3359324122355546, 3.6107784776680404, 3.7040574383772156, 4.130867961260749, 4.133533788303586, 4.899031513688672],
    "Revenue": [41.84298632027783, 34.661119537360234, 39.38300087682567, 37.53984488250128, 32.28453118789761, 30.00113847641735, 22.635401277012628, 25.36502221208036, 19.226970048254086, 20.27967917842273, 13.275828499002512, 18.123991212726547, 11.218294472789265, 10.012867848328882, 12.615181154152336, 10.957731335561812, 6.689122639625872, 9.392968661109095, 5.210162615266291, 4.673642540546473, 0.32862551692664154, 0.8976031867492689, 3.165600007954848, 1.931416028704983, 2.5767822446188833, 4.625689457527259, 0.7899736505077737, 2.313806358173546, 1.292360810760447, 0.9531153124098825, 3.782570135712502, 4.857987801146915, 8.9438232087124, 8.17073493579141, 7.412094028378014, 10.336630624804785, 15.996619968225746, 12.568237393671758, 21.342915741299235, 20.11441346128526, 22.83940550303454, 16.98327873697228, 25.14208222879581, 26.104740406676607, 28.912187929191942, 17.84395651991352, 34.53074273928149, 27.69838334813044, 41.51482194316632]
}

# Load Dataset 1: Ice Cream Selling Data
def load_ice_cream_data():
    return pd.DataFrame(ice_cream_data)

# Load Dataset 2: Synthetic Cosine Wave Data
def create_data():
    X = np.linspace(-1, 2, 60)
    X = X[:, np.newaxis]
    y = np.cos(0.8 * np.pi * X)
    noise = np.random.normal(0, 0.05, size=np.shape(X))
    y = y + noise
    return pd.DataFrame({"X": X.flatten(), "Y": y.flatten()})

# App title
st.markdown("<h1 style='text-align: center;'>📈 Interactive Polynomial Regression Playground</h1>", unsafe_allow_html=True)

# Dataset selection
st.sidebar.header("Dataset Selection")
dataset_name = st.sidebar.selectbox("🔹 Select a dataset:", ["Ice Cream Selling Data", "Synthetic Cosine Wave Data"])

# Load selected dataset
if dataset_name == "Ice Cream Selling Data":
    data = load_ice_cream_data()
    X = data["Temperature"].values.reshape(-1, 1)
    y = data["Revenue"].values
else:
    data = create_data()
    X = data["X"].values.reshape(-1, 1)
    y = data["Y"].values

# Sidebar controls
st.sidebar.header("Adjust Polynomial Degree 📏")
degree = st.sidebar.slider("Degree of Polynomial", 1, 10, 1, 1)

# Train Polynomial Regression model
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X, y)
y_pred_poly = poly_model.predict(X)

# Calculate metrics for Polynomial Regression
mse_poly = mean_squared_error(y, y_pred_poly)
mae_poly = mean_absolute_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

# Dynamic Fit Description
if r2_poly > 0.9:
    fit_description = "Good Fit"
    fit_color = "green"
elif r2_poly > 0.7:
    fit_description = "Slightly Underfit"
    fit_color = "orange"
elif r2_poly > 0.5:
    fit_description = "Underfit"
    fit_color = "red"
elif r2_poly > 0.3:
    fit_description = "Slightly Overfit"
    fit_color = "orange"
else:
    fit_description = "Overfit"
    fit_color = "red"

# Create the main plot
fig = go.Figure()

# Scatter plot of actual data points
fig.add_trace(go.Scatter(
    x=X.flatten(), y=y,
    mode="markers",
    marker=dict(size=8, color="blue"),
    name="Actual Data"
))

# Polynomial Regression curve
if dataset_name == "Ice Cream Selling Data":
    x_plot = np.linspace(-5, 5, 500).reshape(-1, 1)  # Set x-axis range between -5 to +5 for Ice Cream Data
else:
    x_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)  # Default range for Synthetic Data

y_plot_poly = poly_model.predict(x_plot)
fig.add_trace(go.Scatter(
    x=x_plot.flatten(), y=y_plot_poly,
    mode="lines",
    line=dict(color="green", width=3),
    name=f"Polynomial Regression (Degree {degree})"
))

# Customize plot layout
if dataset_name == "Ice Cream Selling Data":
    fig.update_layout(
        title=f"<b>Polynomial Regression: Degree {degree}</b>",
        xaxis_title="Temperature (°C)",
        yaxis_title="Ice Cream Sales (units)",
        template="plotly_white",
        height=500,
        xaxis=dict(range=[-5, 5]),  # Set x-axis range between -5 to +5 for Ice Cream Data
        yaxis=dict(range=[0, 50])    # Set y-axis range between 0 to 50 for Ice Cream Data
    )
else:
    fig.update_layout(
        title=f"<b>Polynomial Regression: Degree {degree}</b>",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        height=500
    )

# Display the main graph
st.plotly_chart(fig)

# Dynamic Fit Description
st.header("Model Fit")
st.markdown(f"**Fit Description:** <span style='color:{fit_color};'>{fit_description}</span>", unsafe_allow_html=True)

# Model Metrics
st.header("Model Metrics")
st.write(f"**Mean Squared Error (MSE):** {mse_poly:.4f}")
st.write(f"**Mean Absolute Error (MAE):** {mae_poly:.4f}")
st.write(f"**R-squared (R²):** {r2_poly:.4f}")

# Residual Plot
st.header("Residual Plot")
residuals_poly = y - y_pred_poly
residual_fig = go.Figure()
residual_fig.add_trace(go.Scatter(
    x=X.flatten(), y=residuals_poly,
    mode="markers",
    marker=dict(size=10, color="green"),
    name="Residuals (Polynomial Regression)"
))
residual_fig.update_layout(
    title="<b>Residual Plot</b>",
    xaxis_title="X",
    yaxis_title="Residuals",
    template="plotly_white",
    height=400
)
st.plotly_chart(residual_fig)

# BIC (Bayesian Information Criterion) Chart
st.header("Bayesian Information Criterion (BIC)")
degrees = np.arange(1, 11)
bic_values = []

for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    n = len(y)
    k = d + 1  # Number of parameters (degree + intercept)
    bic = n * np.log(mse) + k * np.log(n)
    bic_values.append(bic)

# Create BIC plot
bic_fig = go.Figure()
bic_fig.add_trace(go.Scatter(
    x=degrees, y=bic_values,
    mode="lines+markers",
    line=dict(color="purple", width=2),
    marker=dict(size=10, color="purple"),
    name="BIC"
))
bic_fig.update_layout(
    title="<b>BIC vs Polynomial Degree</b>",
    xaxis_title="Degree of Polynomial",
    yaxis_title="BIC Value",
    template="plotly_white",
    height=400
)
st.plotly_chart(bic_fig)

# Outlier Analysis
st.header("Outlier Analysis")
outliers = data[np.abs(residuals_poly) > 2 * np.std(residuals_poly)]
st.write(f"Number of outliers: {len(outliers)}")
st.write(outliers)