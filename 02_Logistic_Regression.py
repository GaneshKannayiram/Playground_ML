import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
import seaborn as sns
import matplotlib.pyplot as plt

# Set page layout
st.set_page_config(page_title="Interactive Logistic Regression", layout="wide")

# Load toy datasets
@st.cache_data
def load_toy_datasets():
    return {
        "Ad Click Prediction": pd.DataFrame({
            "X": np.array([1, 2+3, 3+3, 4+3, 5+3, 6+4, 7+5, 8+6, 9+7, 10+8, 100]),
            "Y": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        }),
        "Exam Pass Prediction": pd.DataFrame({
            "X": np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120]),
            "Y": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        })
    }

# App title
st.markdown("<h1 style='text-align: center;'>📈 Interactive Logistic Regression Playground</h1>", unsafe_allow_html=True)

# Dataset selection
st.sidebar.header("Dataset Selection")
datasets = load_toy_datasets()
uploaded_file = st.sidebar.file_uploader("Upload your own dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded dataset successfully!")
else:
    selected_dataset = st.sidebar.selectbox("🔹 Select a dataset:", list(datasets.keys()))
    data = datasets[selected_dataset]

# Ensure dataset has required columns
if "X" not in data.columns or "Y" not in data.columns:
    st.error("Uploaded dataset must contain 'X' and 'Y' columns.")
    st.stop()

# Compute Logistic Regression Model
X_reshaped = data["X"].values.reshape(-1, 1)
Y = data["Y"].values
log_reg = LogisticRegression().fit(X_reshaped, Y)

# Sidebar controls
st.sidebar.header("Adjust Decision Boundary 📏")
w = st.sidebar.slider("Weight (w)", -5.0, 5.0, log_reg.coef_[0][0], 0.1)
b = st.sidebar.slider("Bias (b)", -10.0, 10.0, log_reg.intercept_[0], 0.1)

# Display logistic regression equation in the sidebar
st.sidebar.markdown(
    f"""
    <div style="display: flex; gap: 10px; margin-top: 20px;">
        <div style="flex: 1; padding: 8px; border-radius: 5px; background-color: #333; border: 1px solid #555; color: white; font-weight: bold; text-align: center;">
            P(Y=1) = 1 / (1 + e^(-({w:.2f}x + {b:.2f})))
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Compute user-defined sigmoid curve
def sigmoid(x):
    return 1 / (1 + np.exp(-(w * x + b)))

X_values = np.linspace(min(data["X"]) - 5, max(data["X"]) + 5, 100)
Y_sigmoid = sigmoid(X_values)

# Create columns for layout
col1, col2 = st.columns([0.65, 0.35], gap = "large", border=True)  # Left column (wider), Right column (narrower)

# Left column: Main plot and metrics
with col1:
    # Plot
    fig = go.Figure()

    # Scatter plot of actual data points
    fig.add_trace(go.Scatter(
        x=data["X"], y=data["Y"],
        mode="markers",
        marker=dict(size=10, color="blue"),
        name="Actual Data"
    ))

    # Sigmoid curve
    fig.add_trace(go.Scatter(
        x=X_values, y=Y_sigmoid,
        mode="lines",
        line=dict(color="red", width=3),
        name="Sigmoid Curve"
    ))

    # Decision boundary
    decision_boundary = -b / w
    fig.add_trace(go.Scatter(
        x=[decision_boundary, decision_boundary], y=[0, 1],
        mode="lines",
        line=dict(color="green", width=2, dash="dash"),
        name="Decision Boundary"
    ))

    # Customize plot layout
    fig.update_layout(
        title=f"<b>Logistic Regression Decision Boundary & Sigmoid Curve</b>",
        xaxis_title="X",
        yaxis_title="Probability",
        template="plotly_white",
        height=500
    )

    # Display the graph
    st.plotly_chart(fig)

    # Model Metrics
    st.header("Model Metrics")
    Y_pred = (sigmoid(data["X"]) >= 0.5).astype(int)
    accuracy = accuracy_score(data["Y"], Y_pred)
    precision = precision_score(data["Y"], Y_pred)
    recall = recall_score(data["Y"], Y_pred)
    f1 = f1_score(data["Y"], Y_pred)
    logloss = log_loss(data["Y"], sigmoid(data["X"]))

    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1-Score:** {f1:.2f}")
    st.write(f"**Log Loss:** {logloss:.2f}")

# Right column: Confusion Matrix, ROC Curve, Precision-Recall Curve
with col2:
    # Confusion Matrix
    st.header("Confusion Matrix")
    cm = confusion_matrix(data["Y"], Y_pred)
    fig_cm, ax = plt.subplots(figsize=(1,1))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig_cm)

    # ROC Curve
    st.header("ROC Curve")
    fpr, tpr, _ = roc_curve(data["Y"], sigmoid(data["X"]))
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random Guess"))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height = 300, width = 300)
    st.plotly_chart(fig_roc)

    # Precision-Recall Curve
    st.header("Precision-Recall Curve")
    precision_curve, recall_curve, _ = precision_recall_curve(data["Y"], sigmoid(data["X"]))
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode="lines", name="Precision-Recall Curve"))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height = 300, width = 300)
    st.plotly_chart(fig_pr)

# Learning section
with st.expander("📘 Understanding the Graph"):
    st.write("""
    - **Blue points**: Actual data (0 or 1).  
    - **Red curve**: Sigmoid function that maps inputs to probabilities.  
    - **Green dashed line**: Decision boundary where probability = 0.5.  
    - **Confusion Matrix**: Shows prediction performance.  
    - **ROC Curve**: Illustrates the trade-off between true positive rate and false positive rate.  
    - **Precision-Recall Curve**: Shows the trade-off between precision and recall.  
    """)

# Interactive Quiz
with st.expander("🧠 Interactive Quiz"):
    st.write("""
    **Question:** What happens to the decision boundary if you increase the weight (w)?  
    - A) Moves to the right  
    - B) Moves to the left  
    - C) Stays the same  
    """)
    answer = st.radio("Your answer:", ["A", "B", "C"])
    if answer == "B":
        st.success("Correct! Increasing the weight moves the decision boundary to the left.")
    else:
        st.error("Incorrect. Try again!")