import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.datasets import make_moons, make_circles, make_classification, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import plot_tree
import graphviz

# Set page layout
st.set_page_config(page_title="Interactive XGBoost Classification Playground", layout="wide")

# Load dataset
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Moons":
        X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
    elif dataset_name == "Circles":
        X, y = make_circles(n_samples=500, noise=0.2, factor=0.5, random_state=42)
    elif dataset_name == "Linear":
        X, y = make_classification(
            n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42
        )
    elif dataset_name == "Iris":
        data = load_iris()
        X, y = data.data[:, :2], data.target  # Use only the first two features for visualization
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data[:, :2], data.target  # Use only the first two features for visualization
    return pd.DataFrame(X, columns=["Feature 1", "Feature 2"]), y

# App title
st.markdown("<h1 style='text-align: center;'>🎯 Interactive XGBoost Classification Playground</h1>", unsafe_allow_html=True)

# Dataset selection
st.sidebar.header("Dataset Selection")
dataset_name = st.sidebar.selectbox("🔹 Select a dataset:", ["Moons", "Circles", "Linear", "Iris", "Breast Cancer"])
X, y = load_data(dataset_name)

# Sidebar controls for hyperparameters
st.sidebar.header("Adjust Hyperparameters 📏")
n_estimators = st.sidebar.slider("Number of Estimators (n_estimators)", 1, 200, 100, 1)
max_depth = st.sidebar.slider("Max Depth of Trees (max_depth)", 1, 10, 3, 1)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
subsample = st.sidebar.slider("Subsample Ratio (subsample)", 0.1, 1.0, 1.0, 0.1)
colsample_bytree = st.sidebar.slider("Column Subsample Ratio (colsample_bytree)", 0.1, 1.0, 1.0, 0.1)
gamma = st.sidebar.slider("Gamma (Minimum Loss Reduction)", 0, 10, 0, 1)
reg_lambda = st.sidebar.slider("L2 Regularization (reg_lambda)", 0, 10, 1, 1)
reg_alpha = st.sidebar.slider("L1 Regularization (reg_alpha)", 0, 10, 0, 1)

# Train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    gamma=gamma,
    reg_lambda=reg_lambda,
    reg_alpha=reg_alpha,
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
fig = go.Figure()

# Scatter plot of actual data points
fig.add_trace(go.Scatter(
    x=X["Feature 1"], y=X["Feature 2"],
    mode="markers",
    marker=dict(size=10, color=y, colorscale="Bluered"),
    name="Actual Data"
))

# Decision boundary contour
fig.add_trace(go.Contour(
    x=xx[0], y=yy[:, 0], z=Z,
    colorscale="Greens",
    opacity=0.5,
    showscale=False,
    name="Decision Boundary"
))

# Customize plot layout
fig.update_layout(
    title="XGBoost Decision Boundary",
    xaxis_title="Feature 1",
    yaxis_title="Feature 2",
    template="plotly_white",
    height=500
)

# Display the graph
st.plotly_chart(fig, use_container_width=True)

# Model Metrics
st.header("Model Metrics")
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average="weighted")
recall = recall_score(y, y_pred, average="weighted")
f1 = f1_score(y, y_pred, average="weighted")

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

# Feature Importance
st.header("Feature Importance")
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
fig_importance = px.bar(feature_importance_df, x="Feature", y="Importance", title="Feature Importance")
st.plotly_chart(fig_importance, use_container_width=True)

# Individual Weak Learners (Decision Boundaries)
st.header("Individual Weak Learners")
st.write("Visualization of the decision boundaries of the first few weak learners:")

# Determine the number of weak learners to visualize
num_learners_to_visualize = min(3, n_estimators)  # Visualize up to 3 weak learners

# Plot decision boundaries of the first few weak learners
if num_learners_to_visualize > 0:
    fig_weak_learners, axes = plt.subplots(1, num_learners_to_visualize, figsize=(6 * num_learners_to_visualize, 6))
    if num_learners_to_visualize == 1:
        axes = [axes]  # Ensure axes is iterable even for a single subplot
    for i in range(num_learners_to_visualize):
        weak_learner = model.get_booster().get_dump()[i]
        xx, yy, Z = plot_decision_boundary(X, y, model)  # Use the full model for simplicity
        axes[i].contourf(xx, yy, Z, alpha=0.5, cmap="Greens")
        axes[i].scatter(X["Feature 1"], X["Feature 2"], c=y, cmap="coolwarm", edgecolor="k")
        axes[i].set_title(f"Weak Learner {i+1}")
    st.pyplot(fig_weak_learners)
else:
    st.write("No weak learners to visualize.")

# First Few Trees in XGBoost
st.header("First Few Trees in XGBoost")
st.write("Visualization of the structure of the first few weak learners:")

# Determine the number of trees to visualize
num_trees_to_visualize = min(3, n_estimators)  # Visualize up to 3 trees

# Plot the first few trees using Graphviz for exact styling
if num_trees_to_visualize > 0:
    for i in range(num_trees_to_visualize):
        # Export the tree to Graphviz
        dot_data = xgb.to_graphviz(model, num_trees=i, rankdir="LR")
        
        # Convert the Graphviz source to a string
        dot_string = str(dot_data)
        
        # Customize the tree style to match AdaBoost
        dot_string = dot_string.replace(
            'digraph {',
            'digraph { node [shape=box, style="filled", color="lightblue"]; edge [fontname="Helvetica", fontsize=10];'
        )
        
        # Render the tree using Graphviz
        st.graphviz_chart(dot_string)
else:
    st.write("No trees to visualize.")

# Misclassified Points
st.header("Misclassified Points")
misclassified_indices = y != y_pred
misclassified = X[misclassified_indices]
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

# Comparison with Default Model
st.header("Comparison with Default Model")
default_model = xgb.XGBClassifier(random_state=42)
default_model.fit(X, y)
default_y_pred = default_model.predict(X)
default_accuracy = accuracy_score(y, default_y_pred)
default_precision = precision_score(y, default_y_pred, average="weighted")
default_recall = recall_score(y, default_y_pred, average="weighted")
default_f1 = f1_score(y, default_y_pred, average="weighted")

st.write(f"**Default Model Accuracy:** {default_accuracy:.2f}")
st.write(f"**Default Model Precision:** {default_precision:.2f}")
st.write(f"**Default Model Recall:** {default_recall:.2f}")
st.write(f"**Default Model F1-Score:** {default_f1:.2f}")