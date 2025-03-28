import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Set page layout to wide
st.set_page_config(page_title="Interactive Lasso Regression", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>🎯 Interactive Lasso Regression Playground</h1>", unsafe_allow_html=True)

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
    }),
    "Feature Selection": pd.DataFrame({
        "X1": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "X2": np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.4, 0.3, 0.5, 0.4, 0.6]),
        "X3": np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
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
elif selected_dataset == "Multicollinear":
    X = data[["X1", "X2"]].values
    feature_names = ["X1", "X2"]
else:  # Feature Selection
    X = data[["X1", "X2", "X3"]].values
    feature_names = ["X1", "X2", "X3"]

y = data["Y"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Alpha controls with presets
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("Show OLS (λ=0)"):
        st.session_state.alpha = 0.0
with col2:
    if st.button("Optimal λ (~0.5)"):
        st.session_state.alpha = 0.5
with col3:
    if st.button("Over-regularized (λ=5)"):
        st.session_state.alpha = 5.0

alpha = st.sidebar.slider(
    "Regularization Strength (λ)", 
    min_value=0.0, 
    max_value=5.0, 
    value=getattr(st.session_state, 'alpha', 0.0), 
    step=0.1,
    help="0 = OLS, 0.1-1 = Good regularization, >2 = Strong feature selection"
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

# Lasso Model
if selected_dataset == "Basic Linear":
    model_lasso = Lasso(alpha=alpha).fit(X_train, y_train)
    coef_lasso = np.array([model_lasso.coef_[0]])
    intercept_lasso = model_lasso.intercept_
else:
    model_lasso = make_pipeline(
        StandardScaler(),
        Lasso(alpha=alpha)
    ).fit(X_train, y_train)
    coef_lasso = model_lasso.named_steps['lasso'].coef_
    intercept_lasso = model_lasso.named_steps['lasso'].intercept_

# Coefficients
coef_ols = model_ols.coef_ if selected_dataset != "Basic Linear" else np.array([model_ols.coef_[0]])

# Predictions
if selected_dataset == "Basic Linear":
    data["Y_pred_ols"] = model_ols.predict(data[["X"]])
    data["Y_pred_lasso"] = model_lasso.predict(data[["X"]])
else:
    data["Y_pred_ols"] = model_ols.predict(data[feature_names])
    data["Y_pred_lasso"] = model_lasso.predict(data[feature_names])

# Metrics
mse_ols_train = np.mean((y_train - model_ols.predict(X_train)) ** 2)
mse_lasso_train = np.mean((y_train - model_lasso.predict(X_train)) ** 2)
mse_ols_test = np.mean((y_test - model_ols.predict(X_test)) ** 2)
mse_lasso_test = np.mean((y_test - model_lasso.predict(X_test)) ** 2)

l1_norm_ols = np.sum(np.abs(coef_ols))
l1_norm_lasso = np.sum(np.abs(coef_lasso))

zero_coef_count = np.sum(coef_lasso == 0)

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
        x=data["X"], y=data["Y_pred_lasso"],
        mode="lines", name=f"Lasso (λ={alpha})",
        line=dict(color="green", width=3)
    ))
    
    if show_residuals:
        for i in range(len(data)):
            fig.add_trace(go.Scatter(
                x=[data["X"].iloc[i], data["X"].iloc[i]],
                y=[data["Y"].iloc[i], data["Y_pred_lasso"].iloc[i]],
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
    
else:  # 3D Plot for multi-feature data
    fig = go.Figure()
    
    # Actual data points
    fig.add_trace(go.Scatter3d(
        x=data[feature_names[0]], 
        y=data[feature_names[1]] if len(feature_names) > 1 else np.zeros(len(data)),
        z=data["Y"],
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
    x_range = np.linspace(data[feature_names[0]].min()-1, data[feature_names[0]].max()+1, 20)
    y_range = np.linspace(data[feature_names[1]].min()-1, data[feature_names[1]].max()+1, 20) if len(feature_names) > 1 else [0, 1]
    xx, yy = np.meshgrid(x_range, y_range)
    
    # OLS surface
    if show_ols and len(feature_names) > 1:
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
    
    # Lasso surface (only for 2 features)
    if len(feature_names) > 1:
        zz_lasso = intercept_lasso + coef_lasso[0]*xx + coef_lasso[1]*yy
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz_lasso,
            colorscale='Greens',
            opacity=0.7,
            name=f"Lasso (λ={alpha})",
            showscale=False,
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True
            )
        ))
    
    # Add dynamic equation annotation
    if len(feature_names) == 2:
        equation_text = f"Lasso: Y = {intercept_lasso:.2f} + {coef_lasso[0]:.2f}{feature_names[0]} + {coef_lasso[1]:.2f}{feature_names[1]}"
    elif len(feature_names) == 3:
        equation_text = f"Lasso: {coef_lasso[0]:.2f}{feature_names[0]} + {coef_lasso[1]:.2f}{feature_names[1]} + {coef_lasso[2]:.2f}{feature_names[2]}"
    else:
        equation_text = f"Lasso: Y = {intercept_lasso:.2f} + {coef_lasso[0]:.2f}{feature_names[0]}"
    
    fig.add_annotation(
        text=equation_text,
        x=0.5, y=1.05,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12))
    
    fig.update_layout(
        scene=dict(
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1] if len(feature_names) > 1 else "",
            zaxis_title="Y (Target)",
            xaxis=dict(range=[data[feature_names[0]].min()-1, data[feature_names[0]].max()+1]),
            yaxis=dict(range=[data[feature_names[1]].min()-1, data[feature_names[1]].max()+1]) if len(feature_names) > 1 else dict(range=[-1, 1]),
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
            "Model": ["OLS", f"Lasso (λ={alpha})"],
            "Coefficient": [f"{coef_ols[0]:.2f}", f"{coef_lasso[0]:.2f}"]
        })
    else:
        coef_table = pd.DataFrame({
            "Feature": feature_names,
            "OLS": [f"{x:.2f}" for x in coef_ols],
            f"Lasso (λ={alpha})": [f"{x:.2f}" for x in coef_lasso]
        })
        
    st.dataframe(coef_table, hide_index=True)
    
    # Key concepts (only for multi-feature)
    if selected_dataset != "Basic Linear":
        with st.expander("📚 Key Concepts"):
            st.markdown(f"""
            ### Lasso Regression Concepts
            
            **1. Regularization (λ)**
            - Current λ value: {alpha}
            - Higher λ → more coefficients become exactly zero
            - Prevents overfitting by feature selection
            
            **2. L1 Penalty**
            - Penalizes absolute coefficient values
            - Can produce exact zeros (feature selection)
            - Particularly useful for high-dimensional data
            
            **3. Feature Selection**
            - Current zero coefficients: {zero_coef_count}/{len(feature_names)}
            - Lasso automatically selects relevant features
            - Eliminates irrelevant features completely
            
            **4. Standardization**
            - {'Enabled' if standardize else 'Disabled'}
            - Critical before applying Lasso
            - Ensures fair penalization of all features
            """)

with col_guide:
    # Performance Metrics section (always shown)
    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("OLS Train MSE", f"{mse_ols_train:.2f}")
    col2.metric("Lasso Train MSE", f"{mse_lasso_train:.2f}")
    improvement = mse_ols_train - mse_lasso_train
    col3.metric("Improvement", 
               f"{abs(improvement):.2f}",
               delta=f"{improvement:.2f}" if improvement != 0 else "0.00")

    col1, col2 = st.columns(2)
    col1.metric("OLS Test MSE", f"{mse_ols_test:.2f}")
    col2.metric("Lasso Test MSE", f"{mse_lasso_test:.2f}")

    col1, col2 = st.columns(2)
    col1.metric("OLS L1 Norm", f"{l1_norm_ols:.2f}")
    col2.metric("Lasso L1 Norm", f"{l1_norm_lasso:.2f}")
    
    if len(feature_names) > 1:
        col1, col2 = st.columns(2)
        col1.metric("Total Features", len(feature_names))
        col2.metric("Zero Coefficients", f"{zero_coef_count}")

    # Diagnostic alerts
    if alpha > 2:
        st.warning(f"⚠️ High λ: {zero_coef_count}/{len(feature_names)} features eliminated (may underfit)")
    elif alpha < 0.1:
        st.info("ℹ️ Low λ: Close to OLS behavior (little feature selection)")
    elif zero_coef_count > 0:
        st.success(f"✅ Feature selection active: {zero_coef_count}/{len(feature_names)} features eliminated")
    
    # Guidance expander
    with st.expander("How to Use the Lasso Regression Visualization", expanded=True):
        st.markdown("""
        <div style="height: 400px; overflow-y: auto;">
        
        #### 1. Understanding Lasso Regression
        
        **Key Difference from Ridge**:
        - Lasso uses L1 regularization (absolute values)
        - Can produce exact zero coefficients (feature selection)
        - Particularly useful for datasets with many features
        
        #### 2. Interacting with the Visualization
        
        **For 2D Plots (Single Feature)**:
        - Observe how the regression line changes with λ
        - Compare with OLS line (red dashed)
        - Enable residuals to see prediction errors
        
        **For 3D Plots (Multiple Features)**:
        - Rotate: Click+drag to view from different angles
        - Zoom: Mouse wheel to zoom in/out
        - Toggle OLS plane to compare with standard regression
        - Watch how the plane flattens as λ increases
        
        #### 3. What to Observe
        
        **As You Increase λ**:
        - Coefficients shrink toward zero
        - Some become exactly zero (feature selection)
        - Model becomes simpler (may underfit if λ too large)
        
        **Performance Metrics**:
        - Watch train/test MSE as you adjust λ
        - Optimal λ often where test MSE is minimized
        - Compare L1 norms (total absolute coefficients)
        
        #### 4. Teaching Points
        
        **λ=0**: 
        - Lasso equals OLS (no regularization)
        
        **Small λ (0.1-1)**: 
        - Some coefficients may become zero
        - Good for feature selection
        
        **Large λ (>2)**: 
        - Most coefficients become zero
        - Risk of underfitting
        
        #### 5. Exploration Tasks
        
        1. Start with λ=0 and compare to OLS
        2. Slowly increase λ to observe feature selection
        3. Find where coefficients first become zero
        4. Identify "sweet spot" with good test MSE
        5. Experiment with different datasets
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
    Interactive Lasso Regression Teaching Tool | Adjust λ to explore feature selection effects
</div>
""", unsafe_allow_html=True)