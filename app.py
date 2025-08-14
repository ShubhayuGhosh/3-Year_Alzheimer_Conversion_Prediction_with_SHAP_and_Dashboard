# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import uuid
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.inspection import partial_dependence
import random

# -------------------------
# Load model & data
# -------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

@st.cache_resource
def load_explainer():
    return pickle.load(open("shap_explainer.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_pickle("baseline_df.pkl")

@st.cache_data
def load_shap_values():
    return np.load("shap_values.npy")

# Load once, cached
model = load_model()
explainer = load_explainer()
X = load_data()
shap_values = load_shap_values()

@st.cache_data
def get_top_features(shap_values, feature_names, top_n=20):
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_mean = np.abs(shap_df).mean().sort_values(ascending=False)
    return shap_mean.index[:top_n]

@st.cache_data
def get_beeswarm_data(shap_values, X, top_feats, sample_size=1000):
    # Sample for speed
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)
        shap_values = shap_values[np.random.choice(len(shap_values), sample_size, replace=False)]
    shap_df = pd.DataFrame(shap_values, columns=top_feats)
    beeswarm_melted = shap_df.melt(var_name="Feature", value_name="SHAP Value")
    beeswarm_melted["Feature Value"] = X[top_feats].melt()["value"]
    return beeswarm_melted

@st.cache_data
def get_shap_interaction_mean(explainer, X_features):
    shap_interaction_values = explainer.shap_interaction_values(X_features)
    return np.abs(shap_interaction_values).mean(axis=0)

# Ensure pseudonyms exist
if "Pseudonym" not in X.columns:
    X["Pseudonym"] = [f"Patient_{uuid.uuid4().hex[:8]}" for _ in range(len(X))]

trained_feature_names = model.booster_.feature_name()
X_features = X[trained_feature_names]

pred_probs = model.predict_proba(X_features)[:, 1]
X["RiskScore"] = pred_probs

# -------------------------
# Streamlit Layout
# -------------------------
st.set_page_config(page_title="🧠 Dementia Risk Dashboard", layout="wide")

tab1, tab2 = st.tabs(["📊 Model Insights", "🧍 Patient Explorer"])

# =========================
# TAB 1: MODEL INSIGHTS
# =========================
with tab1:
    st.title("📊 Model Insights")

    top_feats = get_top_features(shap_values, trained_feature_names)

    # Risk distribution histogram
    st.subheader("📈 Risk Score Distribution")
    with st.expander("ℹ️ What is this?"):
        st.write(
            "This histogram shows how the model’s predicted risk scores are distributed "
            "across all patients. A higher risk score means a higher predicted likelihood of dementia conversion."
        )
    fig_hist = px.histogram(X, x="RiskScore", nbins=20, title="Risk Score Histogram")
    fig_hist.update_layout(xaxis_title="Predicted Risk", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

    if st.checkbox("Show SHAP Beeswarm (Top Features)", value=False):
        with st.expander("ℹ️ What is this?"):
            st.write(
                "A SHAP beeswarm plot shows how each feature impacts the model's predictions "
                "across all patients. Each dot is a patient; color represents the feature's actual value."
            )
        beeswarm_data = get_beeswarm_data(shap_values, X, top_feats)
        fig_beeswarm = px.scatter(
            beeswarm_data,
            x="SHAP Value",
            y="Feature",
            color="Feature Value",
            title="SHAP Beeswarm (Sampled)",
            color_continuous_scale="teal",
            render_mode="webgl"
        )
        fig_beeswarm.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig_beeswarm, use_container_width=True)

    if st.checkbox("Show Partial Dependence Plots", value=False):
        with st.expander("ℹ️ What is this?"):
            st.write(
                "Partial dependence plots (PDPs) show how changing a single feature while "
                "keeping others constant affects the predicted risk. They help interpret non-linear effects."
            )
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for i, feat in enumerate(top_feats[:5]):
            shap.dependence_plot(
                feat, shap_values, X_features, interaction_index=None, ax=axs[i], show=False
            )
        st.pyplot(fig)

    if st.checkbox("Show SHAP Interaction Heatmap", value=False):
        with st.expander("ℹ️ What is this?"):
            st.write(
                "This heatmap shows the average magnitude of interaction effects between pairs of features. "
                "Brighter colors mean stronger interactions, meaning those two features together have a "
                "larger effect on predictions than individually."
            )
        mean_interaction = get_shap_interaction_mean(explainer, X_features)
        fig_heatmap = px.imshow(
            mean_interaction,
            x=trained_feature_names,
            y=trained_feature_names,
            color_continuous_scale="teal",
            title="Mean Absolute SHAP Interaction Values"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


# =========================
# TAB 2: PATIENT EXPLORER
# =========================
with tab2:
    st.title("🧍 Patient Explorer")

    # Search + random selection
    col1, col2 = st.columns([3, 1])
    with col1:
        patient_list = X["Pseudonym"].tolist()
        selected_patient = st.selectbox("Select Patient", patient_list)
    with col2:
        if st.button("🎲 Random Patient"):
            selected_patient = random.choice(patient_list)

    selected_idx = X.index[X["Pseudonym"] == selected_patient][0]

    # SHAP force plot for patient
    st.subheader(f"SHAP Force Plot – {selected_patient}")
    shap_html = shap.force_plot(
        explainer.expected_value,
        shap_values[selected_idx],
        X_features.iloc[selected_idx, :],
        matplotlib=False
    )
    shap_html_path = f"shap_force_{selected_patient}.html"
    shap.save_html(shap_html_path, shap_html)
    with open(shap_html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=300)

    # Similar patients finder
    st.subheader("🧍‍♂️ Similar Patients")
    num_similar = st.slider("Number of similar patients", 1, 10, 5)
    selected_features = X_features.iloc[selected_idx]
    distances = np.linalg.norm(X_features - selected_features, axis=1)
    similar_indices = np.argsort(distances)[1:num_similar + 1]
    similar_table = X.iloc[similar_indices][["Pseudonym", "RiskScore"]]
    st.dataframe(similar_table, use_container_width=True)




