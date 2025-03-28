import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Set page layout to wide
st.set_page_config(page_title="Interactive Ridge Regression", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>🏔️ Interactive Ridge Regression Playground</h1>", unsafe_allow_html=True)

# ========================
# 1. DATASETS
# ========================
datasets = {
    "Basic Linear": pd.DataFrame({
        "X": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "Y": np.array([2, 4, 5, 4, 5, 7, 8, 7, 9, 10])
    }),
    "Multicollinear": pd.DataFrame({
        "X1": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "X2": np.array([1.1, 1.9, 3.2, 3.8, 5.1, 6.2, 6.9, 8.1, 9.2, 9.9]),
        "Y": np.array([2, 4, 5, 4, 5, 7, 8, 7, 9, 10])
    })
}

# ========================
# 2. SIDEBAR CONTROLS
# ========================
st.sidebar.header("Controls 🎛️")
selected_dataset = st.sidebar.selectbox("Dataset", list(datasets.keys()))
data = datasets[selected_dataset].copy()

# Prepare features
if selected_dataset == "Basic Linear":
    X = data[["X"]].values
    feature_names = ["X"]
else:
    X = data[["X1", "X2"]].values
    feature_names = ["X1", "X2"]

y = data["Y"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Alpha controls with presets
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("Show OLS (α=0)"):
        st.session_state.alpha = 0.0
with col2:
    if st.button("Optimal α (~1)"):
        st.session_state.alpha = 1.0
with col3:
    if st.button("Overfit (α=10)"):
        st.session_state.alpha = 10.0

alpha = st.sidebar.slider(
    "Regularization Strength (α)", 
    min_value=0.0, 
    max_value=10.0, 
    value=getattr(st.session_state, 'alpha', 0.0), 
    step=0.1,
    help="0 = OLS, 0.1-1 = Good regularization, >5 = Over-smoothing"
)

standardize = st.sidebar.checkbox(
    "Standardize Features", 
    value=True,
    disabled=(selected_dataset == "Basic Linear")
)

show_ols = st.sidebar.checkbox("Show OLS Line", True)
show_residuals = st.sidebar.checkbox("Show Residuals", False)

# ========================
# 3. MODEL TRAINING
# ========================
# OLS Model
model_ols = LinearRegression().fit(X_train, y_train)

# Ridge Model
if selected_dataset == "Basic Linear":
    model_ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    coef_ridge = np.array([model_ridge.coef_[0]])
    intercept_ridge = model_ridge.intercept_
else:
    model_ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha)
    ).fit(X_train, y_train)
    coef_ridge = model_ridge.named_steps['ridge'].coef_
    intercept_ridge = model_ridge.named_steps['ridge'].intercept_

# Coefficients
coef_ols = model_ols.coef_ if selected_dataset == "Multicollinear" else np.array([model_ols.coef_[0]])

# Predictions
if selected_dataset == "Basic Linear":
    data["Y_pred_ols"] = model_ols.predict(data[["X"]])
    data["Y_pred_ridge"] = model_ridge.predict(data[["X"]])
else:
    data["Y_pred_ols"] = model_ols.predict(data[["X1", "X2"]])
    data["Y_pred_ridge"] = model_ridge.predict(data[["X1", "X2"]])

# Metrics
mse_ols_train = np.mean((y_train - model_ols.predict(X_train)) ** 2)
mse_ridge_train = np.mean((y_train - model_ridge.predict(X_train)) ** 2)
mse_ols_test = np.mean((y_test - model_ols.predict(X_test)) ** 2)
mse_ridge_test = np.mean((y_test - model_ridge.predict(X_test)) ** 2)

l2_norm_ols = np.sqrt(np.sum(coef_ols ** 2))
l2_norm_ridge = np.sqrt(np.sum(coef_ridge ** 2))

# ========================
# 4. MAIN VISUALIZATION
# ========================
if selected_dataset == "Basic Linear":
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["X"], y=data["Y"],
        mode="markers", name="Actual Data",
        marker=dict(size=10, color="blue")
    ))
    
    if show_ols:
        fig.add_trace(go.Scatter(
            x=data["X"], y=data["Y_pred_ols"],
            mode="lines", name="OLS Line",
            line=dict(color="red", dash="dash")
        ))
    
    fig.add_trace(go.Scatter(
        x=data["X"], y=data["Y_pred_ridge"],
        mode="lines", name=f"Ridge (α={alpha})",
        line=dict(color="green", width=3)
    ))
    
    if show_residuals:
        for i in range(len(data)):
            fig.add_trace(go.Scatter(
                x=[data["X"].iloc[i], data["X"].iloc[i]],
                y=[data["Y"].iloc[i], data["Y_pred_ridge"].iloc[i]],
                mode="lines", line=dict(color="gray", width=1),
                showlegend=False
            ))
    
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(range=[0, max(data["Y"])*1.1]),
        height=500,
        margin=dict(l=50, r=50, t=50, b=80)
    )
    
else:  # 3D Plot for multicollinear data
    fig = go.Figure()
    
    # Actual data points
    fig.add_trace(go.Scatter3d(
        x=data["X1"], y=data["X2"], z=data["Y"],
        mode="markers", 
        marker=dict(
            size=5, 
            color=data["Y"],
            colorscale='Viridis',
            opacity=0.8
        ),
        name="Actual Data"
    ))
    
    # Create grid for surfaces
    x_range = np.linspace(data["X1"].min()-1, data["X1"].max()+1, 20)
    y_range = np.linspace(data["X2"].min()-1, data["X2"].max()+1, 20)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # OLS surface
    if show_ols:
        zz_ols = model_ols.intercept_ + model_ols.coef_[0]*xx + model_ols.coef_[1]*yy
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz_ols,
            colorscale='Reds',
            opacity=0.7,
            name="OLS Plane",
            showscale=False,
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True
            )
        ))
    
    # Ridge surface
    zz_ridge = intercept_ridge + coef_ridge[0]*xx + coef_ridge[1]*yy
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz_ridge,
        colorscale='Greens',
        opacity=0.7,
        name=f"Ridge (α={alpha})",
        showscale=False,
        contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True
        )
    ))
    
    # Add dynamic equation annotation
    equation_text = f"Ridge: Y = {intercept_ridge:.2f} + {coef_ridge[0]:.2f}X1 + {coef_ridge[1]:.2f}X2"
    fig.add_annotation(
        text=equation_text,
        x=0.5, y=1.05,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title="X1 (Feature 1)",
            yaxis_title="X2 (Feature 2)",
            zaxis_title="Y (Target)",
            xaxis=dict(range=[data["X1"].min()-1, data["X1"].max()+1]),
            yaxis=dict(range=[data["X2"].min()-1, data["X2"].max()+1]),
            zaxis=dict(range=[0, max(data["Y"])*1.1]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        margin=dict(l=0, r=0, t=30, b=100),
        height=700,
        legend=dict(
            title="<b>LEGEND</b>",
            itemsizing="constant",
            bordercolor="black",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

# Create two-column layout with wider right column
col_viz, col_guide = st.columns([1.5, 1])

with col_viz:
    st.plotly_chart(fig, use_container_width=True)
    
    # Add space below chart
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Coefficient table
    st.subheader("Coefficients")
    if selected_dataset == "Basic Linear":
        coef_table = pd.DataFrame({
            "Model": ["OLS", f"Ridge (α={alpha})"],
            "Coefficient": [f"{coef_ols[0]:.2f}", f"{coef_ridge[0]:.2f}"]
        })
    else:
        coef_table = pd.DataFrame({
            "Feature": feature_names,
            "OLS": [f"{x:.2f}" for x in coef_ols],
            f"Ridge (α={alpha})": [f"{x:.2f}" for x in coef_ridge]
        })
        
    st.dataframe(coef_table, hide_index=True)
    
    # Key concepts (only for multicollinear)
    if selected_dataset == "Multicollinear":
        with st.expander("📚 Key Concepts"):
            st.markdown("""
            ### Ridge Regression Concepts
            
            **1. Regularization (α)**
            - Controls model complexity
            - Higher α → stronger shrinkage of coefficients
            - Prevents overfitting by reducing variance
            
            **2. L2 Penalty**
            - Penalizes large coefficients
            - Shrinks coefficients toward zero (but not exactly zero)
            - Particularly useful for multicollinear data
            
            **3. Standardization**
            - Critical before applying Ridge
            - Ensures features are penalized equally
            - Enabled by default for multi-feature datasets
            
            **4. Bias-Variance Tradeoff**
            - Low α: Low bias, high variance (overfitting risk)
            - High α: High bias, low variance (underfitting risk)
            - Optimal α minimizes test error
            """)

with col_guide:
    # Performance Metrics section (always shown)
    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("OLS Train MSE", f"{mse_ols_train:.2f}")
    col2.metric("Ridge Train MSE", f"{mse_ridge_train:.2f}")
    improvement = mse_ols_train - mse_ridge_train
    col3.metric("Improvement", 
               f"{abs(improvement):.2f}",
               delta=f"{improvement:.2f}" if improvement != 0 else "0.00")

    col1, col2 = st.columns(2)
    col1.metric("OLS Test MSE", f"{mse_ols_test:.2f}")
    col2.metric("Ridge Test MSE", f"{mse_ridge_test:.2f}")

    col1, col2 = st.columns(2)
    col1.metric("OLS L2 Norm", f"{l2_norm_ols:.2f}")
    col2.metric("Ridge L2 Norm", f"{l2_norm_ridge:.2f}")

    # Diagnostic alerts
    if alpha > 5:
        st.warning("⚠️ High α: Model may be underfitting (check test MSE)")
    elif alpha < 0.1:
        st.info("ℹ️ Low α: Close to OLS behavior")
    
    # Guidance expander (only for multicollinear)
    if selected_dataset == "Multicollinear":
        with st.expander("How to Use the 3D Ridge Regression Visualization", expanded=True):
            st.markdown("""
            <div style="height: 400px; overflow-y: auto;">
            
            #### 1. Understanding the Components
            - **Blue Points**: Actual data points (X1, X2, Y) plotted in 3D space
            - **Red Surface (OLS Plane)**: The ordinary least squares regression plane
            - **Green Surface (Ridge Plane)**: Changes with α (always visible)
            
            #### 2. Key Interactions
            - **Adjust α**: Slide from 0 to 10
              - α=0: Ridge matches OLS (surfaces overlap)
              - α increases: Green surface flattens (coefficients shrink)
            - **Rotate**: Click+drag to view from different angles
            - **Zoom**: Mouse wheel to zoom in/out
            - **Toggle OLS**: Compare with standard regression
            
            #### 3. What to Observe
            - **Coefficient Shrinkage**: Green plane becomes less steep
            - **Multicollinearity Effect**: Ridge stabilizes the plane
            - **Residuals**: Enable to see vertical lines (points to plane)
            
            #### 4. Teaching Points
            - **α=0**: Ridge equals OLS (no regularization)
            - **Small α (0.1-1)**: Slight flattening starts
            - **Large α (5-10)**: Extreme flattening (underfitting)
            - **Optimal α**: Where test MSE is minimized
            
            #### 5. Exploration Tasks
            1. Set α=0 and rotate to see Ridge=OLS
            2. Slowly increase α to observe flattening
            3. Find where surfaces diverge (α > 0)
            4. Identify underfitting (α too large)
            5. Locate "sweet spot" with lowest test MSE
            </div>
            """, unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: gray;
    text-align: center;
}
</style>
<div class="footer">
    Interactive Ridge Regression Teaching Tool | Adjust α to explore regularization effects
</div>
""", unsafe_allow_html=True)