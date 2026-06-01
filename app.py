# python -m streamlit run app.py --server.fileWatcherType none
import os
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score
)
from nltk_processor import process_explanation
from langchain_agent import build_agent, run_agent
from shap_explainer import (
    plot_feature_importance,
    get_feature_importance,
    explain_single_prediction
)
# -----------------------------
# Basic checks
# -----------------------------
if not os.path.exists("models/best_model.pkl"):
    st.error("Models not found. Please run phase1.py and phase2.py first.")
    st.stop()
# -----------------------------
# Load saved objects
# -----------------------------
@st.cache_resource
def load_essentials():
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    best_model = joblib.load("models/best_model.pkl")
    best_name = open("models/best_model_name.txt").read().strip()

    stacking = None
    if os.path.exists("models/stacking_model.pkl"):
        stacking = joblib.load("models/stacking_model.pkl")

    return scaler, encoders, feature_names, best_model, best_name, stacking



@st.cache_resource
def load_all_models():
    dt = joblib.load("models/decision_tree.pkl")
    lr = joblib.load("models/logistic_regression.pkl")
    mlp = joblib.load("models/neural_network.pkl")
    stacking = joblib.load("models/stacking_model.pkl")
    X_test, y_test = joblib.load("models/test_data.pkl")

    # Reset indexes so X_test and y_test match correctly
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Use a smaller sample so Streamlit does not freeze
    if len(X_test) > 30000:
        sample_index = X_test.sample(n=30000, random_state=42).index
        X_test = X_test.loc[sample_index]
        y_test = y_test.loc[sample_index]

    return dt, lr, mlp, stacking, X_test, y_test

@st.cache_resource
def load_agent():
    return build_agent()


scaler, encoders, feature_names, best_model, best_name, stacking = load_essentials()


def encode_value(column_name, selected_value):
    """
    Encode dropdown value using the same LabelEncoder used during training.
    Since values come from dropdowns, they are guaranteed to exist in encoder classes.
    """
    return int(encoders[column_name].transform([selected_value])[0])


def predict_transaction(model, row):
    """
    Prepare input, scale it, and predict fraud probability.
    A custom threshold is used because fraud detection needs high recall.
    """
    X = pd.DataFrame([row])[feature_names]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)

    fraud_prob = model.predict_proba(X_scaled)[0][1]

    threshold = 0.30
    prediction = 1 if fraud_prob >= threshold else 0

    return fraud_prob, prediction


def get_risk_label(prob):
    if prob < 0.30:
        return "🟢 LOW RISK"
    elif prob < 0.50:
        return "🟡 MEDIUM RISK"
    elif prob < 0.70:
        return "🟠 HIGH RISK"
    else:
        return "🔴 CRITICAL RISK"


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.caption(f"Active model: **{best_name}** — saved model loaded from disk")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💳 Check Transaction",
    "📊 Model Evaluation",
    "📈 Data Insights",
    "🔍 Feature Importance",
    "📄 Fraud Reports",
    "🤖 Fraud Assistant"
])


# ==========================================================
# TAB 1 — CHECK TRANSACTION
# ==========================================================
with tab1:
    st.subheader("Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=1.0,
            value=150.0
        )

        category = st.selectbox(
            "Merchant Category",
            list(encoders["category"].classes_)
        )

        gender = st.selectbox(
            "Cardholder Gender",
            list(encoders["gender"].classes_)
        )
        merchant = st.text_input("Merchant Name", value="Amazon")
        job = st.text_input("Cardholder Job", value="Student")

    with col2:
        hour = st.slider(
            "Hour of Transaction",
            min_value=0,
            max_value=23,
            value=14
        )

        state = st.selectbox(
            "State",
            list(encoders["state"].classes_)
        )

        city = st.text_input("City", value="Bangalore")

        age = st.slider(
            "Cardholder Age",
            min_value=18,
            max_value=90,
            value=35
        )

        city_pop = st.number_input(
            "City Population",
            min_value=100,
            value=50000
        )

    phase = st.radio(
        "Model Phase",
        ["Phase 1 — Best Individual", "Phase 2 — Stacking"],
        horizontal=True
    )

    if "Phase 2" in phase and stacking is not None:
        active_model = stacking
    else:
        active_model = best_model

    if st.button("🔍 Check Transaction", use_container_width=True):
        row = {name: 0 for name in feature_names}
        row["amt"] = amount
        row["category"] = encode_value("category", category)
        row["gender"] = encode_value("gender", gender)
        row["state"] = encode_value("state", state)
        row["city_pop"] = city_pop
        row["hour"] = hour
        row["age"] = age
        row["is_night"] = 1 if hour < 6 else 0
        row["is_high_amount"] = 1 if amount > 500 else 0
        row["is_small_city"] = 1 if city_pop < 5000 else 0

        prob, pred = predict_transaction(active_model, row)
        # -----------------------------
        # Rule-Based Fraud Layer
        # -----------------------------
        rule_score = 0
        rule_reasons = []

        if amount > 1000:
            rule_score += 1
            rule_reasons.append("Transaction amount exceeds typical fraud range")

        if hour < 6:
            rule_score += 1
            rule_reasons.append("Late-night transaction")

        if category in ["shopping_net", "misc_net", "grocery_pos"]:
            rule_score += 1
            rule_reasons.append("High-risk merchant category")

        # Rules increase risk, but final decision still uses model probability
        if rule_score == 1:
            prob = min(prob + 0.10, 1.0)
        elif rule_score == 2:
            prob = min(prob + 0.20, 1.0)
        elif rule_score == 3:
            prob = min(prob + 0.30, 1.0)

        pred = 1 if prob >= 0.30 else 0
        risk = get_risk_label(prob)

        st.markdown("---")

        if pred == 1:
            st.error("## 🚨 FRAUD DETECTED")
        else:
            st.success("## ✅ LEGITIMATE")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraud Probability", f"{prob * 100:.1f}%")
        c2.metric("Risk Score", risk)
        c3.metric("Amount", f"${amount:,.2f}")
        c4.metric("Hour", f"{hour:02d}:00")
        
        st.progress(float(prob))

        st.subheader("📝 GPT-2 Explanation")

        with st.spinner("Generating explanation..."):
            from gpt2_explainer import generate_fraud_explanation

            location_flag = 1 if city_pop < 5000 else 0
            raw_exp = generate_fraud_explanation(prob, amount, hour, location_flag)
            processed_exp = process_explanation(raw_exp)

        explanation_text = processed_exp["clean"]
        
        if rule_reasons:
            explanation_text += (
                " Key risk factors include: "
                + ", ".join(rule_reasons)
                + "."
            )

        if not explanation_text:
            explanation_text = (
                "This transaction was analysed using amount, time, merchant category, "
                "location, and customer details."
            )

        st.info(explanation_text)

        if processed_exp["keywords"]:
            st.caption(
                "🔑 Fraud keywords: "
                + " · ".join(f"`{word}`" for word in processed_exp["keywords"])
            )

        st.subheader("🔍 Feature Contributions")
        contrib_df = explain_single_prediction(row, feature_names)
        st.dataframe(contrib_df, use_container_width=True)

        st.subheader("Recommended Action")

        if prob >= 0.70:
            st.error("🚫 Block this transaction and contact the cardholder immediately.")
        elif prob >= 0.30:
            st.warning("⚠️ Flag this transaction for manual review or OTP verification.")
        else:
            st.success("✅ Approve the transaction.")

        st.session_state["last_transaction"] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "category": category,
            "merchant": merchant,
            "hour": hour,
            "state": state,
            "city": city,
            "age": age,
            "gender": gender,
            "job": job,
            "city_pop": city_pop,
            "fraud_prob": f"{prob * 100:.1f}%",
            "risk": risk,
            "verdict": "FRAUD" if pred == 1 else "LEGITIMATE",
            "model": phase,
            "explanation": explanation_text
        }


# ==========================================================
# TAB 2 — MODEL EVALUATION
# ==========================================================
with tab2:
    st.subheader("Model Performance on Test Data")
    st.info("Click the button below to load evaluation charts.")

    if st.button("Load Model Evaluation", use_container_width=True):
        dt, lr, mlp, stacking_eval, X_test, y_test = load_all_models()

        model_list = [
            ("Decision Tree", dt),
            ("Logistic Regression", lr),
            ("Neural Network", mlp),
            ("Stacking Ensemble", stacking_eval)
        ]

        rows = []

        for name, model in model_list:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            rows.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                "Precision": round(precision_score(y_test, y_pred), 3),
                "Recall": round(recall_score(y_test, y_pred), 3),
                "F1 Score": round(f1_score(y_test, y_pred), 3),
                "ROC AUC": round(roc_auc_score(y_test, y_prob), 3)
            })

        metrics_df = pd.DataFrame(rows)

        st.subheader("📋 Model Comparison Table")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("📊 Model Comparison Chart")

        chart_df = metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]]
        st.bar_chart(chart_df)

        selected_model_name = st.selectbox(
            "Select model for detailed view",
            metrics_df["Model"].tolist()
        )

        selected_model = dict(model_list)[selected_model_name]

        y_pred_selected = selected_model.predict(X_test)
        y_prob_selected = selected_model.predict_proba(X_test)[:, 1]

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(
                confusion_matrix(y_test, y_pred_selected),
                display_labels=["Legit", "Fraud"]
            ).plot(ax=ax, colorbar=False)

            ax.set_title(selected_model_name)
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.subheader("ROC Curve")

            fig, ax = plt.subplots()

            for name, model in model_list:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                ax.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.2f}")

            ax.plot([0, 1], [0, 1], "--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(fontsize=7)

            st.pyplot(fig)
            plt.close(fig)


# ==========================================================
# TAB 3 — DATA INSIGHTS
# ==========================================================
with tab3:
    st.subheader("Dataset Visualizations")
    st.info("Click the button below to load sample dataset charts.")

    if st.button("Load Data Insights", use_container_width=True):
        data_path = "data/fraudTrain.csv"

        if not os.path.exists(data_path):
            st.error("fraudTrain.csv not found inside the data folder.")
        else:
            df = pd.read_csv(data_path, nrows=30000)
            df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour

            st.write(f"Sample loaded: **{len(df):,} transactions**")
            st.write(f"Fraud rate in sample: **{df['is_fraud'].mean() * 100:.2f}%**")

            st.dataframe(
                df[[
                    "trans_date_trans_time",
                    "merchant",
                    "category",
                    "amt",
                    "gender",
                    "city",
                    "state",
                    "is_fraud"
                ]].head(10),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Fraud vs Legitimate")
                fraud_counts = df["is_fraud"].value_counts().rename(index={0: "Legit", 1: "Fraud"})
                st.bar_chart(fraud_counts)

            with col2:
                st.subheader("Fraud Rate by Hour")
                hourly_fraud = df.groupby("hour")["is_fraud"].mean()
                st.line_chart(hourly_fraud)

            st.subheader("Fraud Rate by Category")
            category_fraud = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False)
            st.bar_chart(category_fraud)


# ==========================================================
# TAB 4 — FEATURE IMPORTANCE
# ==========================================================
with tab4:
    st.subheader("Feature Importance Analysis")
    st.markdown("This shows which features the Decision Tree used most while detecting fraud.")

    if st.button("Show Feature Importance", use_container_width=True):
        fig = plot_feature_importance(top_n=15)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Top 5 Features:**")
        top5 = get_feature_importance(top_n=5)
        st.table(top5)

        st.info("""
        Usually important fraud signals include:
        - Transaction amount
        - Merchant category
        - Transaction hour
        - City population
        - Merchant behavior
        """)


# ==========================================================
# TAB 5 — FRAUD REPORT
# ==========================================================
with tab5:
    st.subheader("Download Investigation Report")
    st.markdown("First check a transaction in Tab 1. Then download the report here.")

    if "last_transaction" not in st.session_state:
        st.info("No transaction analysed yet.")
    else:
        t = st.session_state["last_transaction"]

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Timestamp:** {t['timestamp']}")
            st.write(f"**Amount:** ${t['amount']:,.2f}")
            st.write(f"**Merchant:** {t['merchant']}")
            st.write(f"**Category:** {t['category']}")
            st.write(f"**Hour:** {t['hour']:02d}:00")

        with col2:
            st.write(f"**Fraud Probability:** {t['fraud_prob']}")
            st.write(f"**Risk Level:** {t['risk']}")
            st.write(f"**Verdict:** {t['verdict']}")
            st.write(f"**Model:** {t['model']}")

        report = f"""
CREDIT CARD FRAUD DETECTION — INVESTIGATION REPORT
Generated: {t['timestamp']}

TRANSACTION DETAILS
Amount: ${t['amount']:,.2f}
Merchant: {t['merchant']}
Category: {t['category']}
Hour: {t['hour']:02d}:00
Location: {t['city']}, {t['state']}
Cardholder: Age {t['age']}, Gender {t['gender']}, Job {t['job']}
City Population: {t['city_pop']}

FRAUD ANALYSIS
Fraud Probability: {t['fraud_prob']}
Risk Level: {t['risk']}
Verdict: {t['verdict']}
Model Used: {t['model']}

AI EXPLANATION
{t['explanation']}

RECOMMENDED ACTION
{"Block / verify this transaction immediately." if t['verdict'] == "FRAUD" else "Approve this transaction."}
        """.strip()

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


# ==========================================================
# TAB 6 — FRAUD ASSISTANT
# ==========================================================
with tab6:
    st.subheader("🤖 LangChain Fraud Assistant")
    st.markdown("Ask a simple fraud-related question.")

    query = st.text_input(
        "Describe a transaction:",
        placeholder="Example: Is a $2500 transaction at 3AM suspicious?"
    )

    if st.button("Ask Agent", use_container_width=True):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Agent analysing..."):
                agent = load_agent()
                answer = run_agent(agent, query)

            st.markdown("**Agent Response:**")
            st.write(answer)

    st.markdown("---")
    st.markdown("""
    **How the assistant works:**
    - Uses the saved ML model for prediction
    - Uses GPT-2 for plain-English explanation
    - Uses NLTK for keyword extraction
    """)