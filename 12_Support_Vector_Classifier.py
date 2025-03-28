import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Set page layout
st.set_page_config(page_title="Interactive SVM Playground", layout="wide")

# Load dataset
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Moons":
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
    elif dataset_name == "Circles":
        X, y = make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=42)
    elif dataset_name == "Linear":
        X, y = make_classification(
            n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42
        )
    return pd.DataFrame(X, columns=["Feature 1", "Feature 2"]), y

# App title
st.markdown("<h1 style='text-align: center;'>📊 Interactive SVM Playground</h1>", unsafe_allow_html=True)

# Dataset selection
st.sidebar.header("Dataset Selection")
dataset_name = st.sidebar.selectbox("🔹 Select a dataset:", ["Moons", "Circles", "Linear"])
X, y = load_data(dataset_name)

# Sidebar controls
st.sidebar.header("Adjust Hyperparameters 📏")
C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf"])
gamma = st.sidebar.slider("Gamma (for RBF kernel)", 0.01, 10.0, 1.0, 0.01) if kernel == "rbf" else "scale"
degree = st.sidebar.slider("Degree (for polynomial kernel)", 2, 5, 3) if kernel == "poly" else 3

# Train SVM model
model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree)
model.fit(X, y)

# Create a meshgrid for decision boundary visualization
def plot_decision_boundary(X, y, model):
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

xx, yy, Z = plot_decision_boundary(X, y, model)

# Plot decision boundary and data points
fig = px.scatter(X, x="Feature 1", y="Feature 2", color=y.astype(str), title="SVM Decision Boundary")
fig.update_traces(marker=dict(size=10))
fig.add_contour(x=xx[0], y=yy[:, 0], z=Z, colorscale="Blues", opacity=0.5, showscale=False)
st.plotly_chart(fig, use_container_width=True)

# Model Metrics
st.header("Model Metrics")
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1-Score:** {f1:.2f}")

# Confusion Matrix
st.header("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig_cm)

# Classification Report
st.header("Classification Report")
report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Support Vectors
st.header("Support Vectors")
support_vectors = model.support_vectors_
st.write(f"Number of support vectors: {len(support_vectors)}")
st.write(support_vectors)

# Learning section
with st.expander("📘 Understanding SVM"):
    st.write("""
    - **Decision Boundary**: The line that separates the classes.
    - **Support Vectors**: The data points closest to the decision boundary.
    - **Kernel**: A function used to transform the data into a higher-dimensional space.
    - **C (Regularization)**: Controls the trade-off between maximizing the margin and minimizing classification error.
    - **Gamma**: Controls the influence of individual training samples (for RBF kernel).
    - **Degree**: Controls the degree of the polynomial (for polynomial kernel).
    """)

# Interactive Quiz
with st.expander("🧠 Interactive Quiz"):
    st.write("""
    **Question:** What happens to the decision boundary if you increase the value of C?  
    - A) It becomes more flexible and fits the training data better.  
    - B) It becomes less flexible and fits the training data worse.  
    - C) It remains unchanged.  
    """)
    answer = st.radio("Your answer:", ["A", "B", "C"])
    if answer == "A":
        st.success("Correct! Increasing C makes the decision boundary more flexible.")
    else:
        st.error("Incorrect. Try again!")