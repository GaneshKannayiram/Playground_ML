import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
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
st.set_page_config(page_title="Interactive Random Forest Playground", layout="wide")

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
st.markdown("<h1 style='text-align: center;'>🌲 Interactive Random Forest Playground</h1>", unsafe_allow_html=True)

# Dataset selection
st.sidebar.header("Dataset Selection")
dataset_name = st.sidebar.selectbox("🔹 Select a dataset:", ["Moons", "Circles", "Linear"])
X, y = load_data(dataset_name)

# Sidebar controls
st.sidebar.header("Adjust Hyperparameters 📏")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 1, 100, 10, 1)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5, 1)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, 1)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1, 1)
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    random_state=42,
)
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
fig = px.scatter(X, x="Feature 1", y="Feature 2", color=y.astype(str), title="Random Forest Decision Boundary")
fig.update_traces(marker=dict(size=10))
fig.add_contour(x=xx[0], y=yy[:, 0], z=Z, colorscale="Greens", opacity=0.5, showscale=False)
st.plotly_chart(fig, use_container_width=True)

# Visualize a few individual trees
st.header("Individual Trees in the Forest")
fig_trees, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, ax in enumerate(axes):
    plot_tree(model.estimators_[i], filled=True, feature_names=X.columns, class_names=["0", "1"], ax=ax)
st.pyplot(fig_trees)

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

# Comparison with Default Model
st.header("Comparison with Default Model")
default_model = RandomForestClassifier(random_state=42)
default_model.fit(X, y)
default_y_pred = default_model.predict(X)
default_accuracy = accuracy_score(y, default_y_pred)

st.write(f"**Default Model Accuracy:** {default_accuracy:.2f}")
st.write(f"**Your Model Accuracy:** {accuracy:.2f}")

# Misclassified Points
st.header("Misclassified Points")
misclassified = X[y != y_pred]
st.write(f"Number of misclassified points: {len(misclassified)}")
st.write(misclassified)

# Plot misclassified points
if not misclassified.empty:
    fig_misclassified = px.scatter(
        X, x="Feature 1", y="Feature 2", color=y.astype(str),
        title="Misclassified Points"
    )
    fig_misclassified.add_trace(
        px.scatter(misclassified, x="Feature 1", y="Feature 2", color_discrete_sequence=["red"]).data[0]
    )
    st.plotly_chart(fig_misclassified, use_container_width=True)