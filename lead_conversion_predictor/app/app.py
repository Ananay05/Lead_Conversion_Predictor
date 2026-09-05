"""
app.py — Lead Conversion Prediction Web App (Streamlit)

Author: [Your Name]
College: [Your College Name]
Subject: Machine Learning / Data Science Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from utils.predict_utils import predict, get_model_results
from utils.db_utils import init_db, save_prediction, fetch_all_predictions, fetch_summary_stats

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lead Conversion Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Init DB ───────────────────────────────────────────────────────────────────
init_db()

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 5px solid #4361ee;
    }
    .result-box-yes {
        background: #d4edda;
        border-left: 6px solid #28a745;
        border-radius: 10px;
        padding: 1.2rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #155724;
    }
    .result-box-no {
        background: #f8d7da;
        border-left: 6px solid #dc3545;
        border-radius: 10px;
        padding: 1.2rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #721c24;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/goal.png", width=64)
    st.markdown("## 🎯 Lead Predictor")
    st.markdown("---")
    page = st.radio("Navigate", ["🔮 Predict", "📊 Dashboard", "🗄️ History", "📈 Model Info"])
    st.markdown("---")
    st.markdown("""
    **About this project:**  
    Built as a college ML project using:
    - Logistic Regression
    - Decision Tree
    - Streamlit + SQLite
    - Pandas / NumPy / Sklearn
    """)
    st.markdown('<div class="footer">Made with ❤️ for college project</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.markdown('<div class="main-title">🎯 Lead Conversion Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Fill in lead details below and predict the probability of conversion.</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Lead Profile")
        age = st.slider("Age", 18, 65, 30)
        income = st.number_input("Annual Income (₹)", min_value=20000, max_value=1500000, value=500000, step=10000)
        industry = st.selectbox("Industry", ["Tech", "Finance", "Healthcare", "Education", "Retail"])
        lead_source = st.selectbox("Lead Source", ["Organic", "Paid Ad", "Referral", "Social Media", "Email"])

    with col2:
        st.markdown("#### 🌐 Website Behaviour")
        website_visits = st.slider("Website Visits", 1, 20, 5)
        time_spent = st.slider("Time Spent on Site (mins)", 0.0, 30.0, 5.0, step=0.5)
        pages_viewed = st.slider("Pages Viewed", 1, 15, 4)

    with col3:
        st.markdown("#### 📧 Engagement")
        email_opened = st.radio("Opened Email?", ["No", "Yes"])
        prev_interaction = st.radio("Had Previous Interaction?", ["No", "Yes"])
        lead_score = st.slider("Lead Score (0–100)", 0, 100, 50)
        follow_up_calls = st.slider("Follow-up Calls Made", 0, 5, 1)

    st.markdown("---")

    if st.button("🔮 Predict Conversion", use_container_width=True):
        input_data = {
            "age": age,
            "income": income,
            "lead_source": lead_source,
            "website_visits": website_visits,
            "time_spent_on_site": time_spent,
            "pages_viewed": pages_viewed,
            "email_opened": 1 if email_opened == "Yes" else 0,
            "previous_interaction": 1 if prev_interaction == "Yes" else 0,
            "lead_score": lead_score,
            "industry": industry,
            "follow_up_calls": follow_up_calls,
        }

        with st.spinner("Running prediction..."):
            prediction, probability = predict(input_data, model_choice)

        save_prediction(input_data, model_choice, prediction, probability)

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1:
                st.markdown(f'<div class="result-box-yes">✅ Will Convert &nbsp;&nbsp; Probability: {probability:.1%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box-no">❌ Won\'t Convert &nbsp;&nbsp; Probability: {probability:.1%}</div>', unsafe_allow_html=True)

        with res_col2:
            fig, ax = plt.subplots(figsize=(3.5, 0.6))
            color = "#28a745" if prediction == 1 else "#dc3545"
            ax.barh([""], [probability], color=color, height=0.4)
            ax.barh([""], [1 - probability], left=[probability], color="#e0e0e0", height=0.4)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            ax.set_title("Conversion Probability", fontsize=9, pad=4)
            st.pyplot(fig)
            plt.close()

        st.success("✅ Prediction saved to database!")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown('<div class="main-title">📊 Predictions Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time overview of all predictions made so far.</div>', unsafe_allow_html=True)

    stats = fetch_summary_stats()
    df_all = fetch_all_predictions()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", stats["total_predictions"])
    c2.metric("Predicted Conversions", stats["predicted_conversions"])
    c3.metric("Avg Probability", f"{stats['avg_probability']:.2%}" if stats['avg_probability'] else "—")

    if len(df_all) == 0:
        st.info("No predictions yet. Go to the Predict page to get started!")
    else:
        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Conversion Distribution")
            fig, ax = plt.subplots()
            counts = df_all["prediction"].value_counts()
            labels = ["Not Converted", "Converted"]
            colors = ["#ef233c", "#06d6a0"]
            ax.pie(counts.values, labels=[labels[i] for i in counts.index],
                   colors=[colors[i] for i in counts.index],
                   autopct="%1.1f%%", startangle=90)
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("##### Probability Distribution")
            fig, ax = plt.subplots()
            ax.hist(df_all["probability"], bins=20, color="#4361ee", edgecolor="white")
            ax.set_xlabel("Probability")
            ax.set_ylabel("Count")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close()

        st.markdown("##### Model Usage")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        model_counts = df_all["model_used"].value_counts()
        ax.barh(model_counts.index, model_counts.values, color=["#4361ee", "#f72585"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Times Used")
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️ History":
    st.markdown('<div class="main-title">🗄️ Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">All predictions stored in SQLite database.</div>', unsafe_allow_html=True)

    df_all = fetch_all_predictions()

    if len(df_all) == 0:
        st.info("No prediction history found yet.")
    else:
        st.dataframe(df_all, use_container_width=True)
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download as CSV", csv, "prediction_history.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Info":
    st.markdown('<div class="main-title">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation metrics from training. Models trained on synthetic lead data.</div>', unsafe_allow_html=True)

    results = get_model_results()

    for model_name, res in results.items():
        with st.expander(f"📌 {model_name}", expanded=True):
            m1, m2 = st.columns(2)
            m1.metric("Accuracy",  f"{res['accuracy']:.2%}")
            m2.metric("ROC-AUC",   f"{res['roc_auc']:.4f}")

            st.markdown("**Classification Report:**")
            st.code(res["report"])

            st.markdown("**Confusion Matrix:**")
            cm = np.array(res["confusion"])
            fig, ax = plt.subplots(figsize=(3, 2.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Not Conv.", "Conv."])
            ax.set_yticklabels(["Not Conv.", "Conv."])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            fig.colorbar(im, ax=ax, shrink=0.8)
            st.pyplot(fig)
            plt.close()
