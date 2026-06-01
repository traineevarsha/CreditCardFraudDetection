# langchain_agent.py
# Simple LangChain-style fraud assistant
# Uses LangChain Tool objects, but keeps the logic easy to understand.

from langchain.tools import Tool
import joblib
import pandas as pd
import re

from gpt2_explainer import generate_fraud_explanation
from nltk_processor import process_explanation


FRAUD_THRESHOLD = 0.30


def predict_fraud_tool(input_str: str) -> str:
    """
    Predict fraud from amount and hour.
    Input format: "amount,hour"
    """

    try:
        parts = [float(x.strip()) for x in input_str.split(",")]

        amount = parts[0]
        hour = parts[1] if len(parts) > 1 else 12.0

        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")

        row = {name: 0 for name in feature_names}

        if "amt" in feature_names:
            row["amt"] = amount

        if "hour" in feature_names:
            row["hour"] = hour

        if "is_night" in feature_names:
            row["is_night"] = 1 if hour < 6 else 0

        if "is_high_amount" in feature_names:
            row["is_high_amount"] = 1 if amount > 500 else 0

        X = pd.DataFrame([row])[feature_names]
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)

        prob = model.predict_proba(X_scaled)[0][1]

        label = "FRAUD" if prob >= FRAUD_THRESHOLD else "LEGITIMATE"

        return f"{label} — Fraud probability: {prob * 100:.1f}%"

    except Exception as e:
        return f"Error: {str(e)}"


def explain_fraud_tool(input_str: str) -> str:
    """
    Generate explanation using GPT-2.
    Input format: "fraud_prob,amount,hour,location_flag"
    """

    try:
        parts = [float(x.strip()) for x in input_str.split(",")]

        fraud_prob = parts[0]
        amount = parts[1] if len(parts) > 1 else 100.0
        hour = int(parts[2]) if len(parts) > 2 else 12
        location_flag = int(parts[3]) if len(parts) > 3 else 0

        return generate_fraud_explanation(
            fraud_prob,
            amount,
            hour,
            location_flag
        )

    except Exception as e:
        return f"Error: {str(e)}"


def extract_keywords_tool(text: str) -> str:
    """
    Extract fraud keywords using NLTK.
    """

    result = process_explanation(text)
    keywords = result["keywords"]
    clean_text = result["clean"]

    if keywords:
        return f"Clean: {clean_text} | Keywords: {', '.join(keywords)}"

    return f"Clean: {clean_text} | Keywords: none"


tools = [
    Tool(
        name="FraudPredictor",
        func=predict_fraud_tool,
        description="Predicts fraud probability. Input: 'amount,hour'"
    ),
    Tool(
        name="FraudExplainer",
        func=explain_fraud_tool,
        description="Uses GPT-2 to explain fraud risk. Input: 'fraud_prob,amount,hour,location_flag'"
    ),
    Tool(
        name="KeywordExtractor",
        func=extract_keywords_tool,
        description="Extracts fraud keywords from explanation text."
    )
]


class FraudAgent:
    """
    Simple custom agent.
    It calls tools in this order:
    prediction -> GPT-2 explanation -> keyword extraction -> recommendation
    """

    def invoke(self, inputs: dict) -> dict:
        query = inputs.get("input", "")

        numbers = re.findall(r"\d+\.?\d*", query)

        amount = float(numbers[0]) if len(numbers) > 0 else 500.0
        hour = float(numbers[1]) if len(numbers) > 1 else 12.0

        if hour > 23:
            hour = 12.0

        prediction = predict_fraud_tool(f"{amount},{hour}")

        try:
            prob_text = prediction.split("Fraud probability: ")[1]
            prob = float(prob_text.replace("%", "")) / 100
        except Exception:
            prob = 0.5

        location_flag = 1 if hour < 6 else 0

        explanation = explain_fraud_tool(
            f"{prob},{amount},{int(hour)},{location_flag}"
        )

        keywords = extract_keywords_tool(explanation)

        if prob >= 0.70:
            recommendation = "BLOCK this transaction immediately."
        elif prob >= FRAUD_THRESHOLD:
            recommendation = "FLAG this transaction for manual review."
        else:
            recommendation = "APPROVE this transaction."

        final_answer = (
            f"Prediction: {prediction}\n\n"
            f"Explanation using GPT-2: {explanation}\n\n"
            f"NLTK Keywords: {keywords}\n\n"
            f"Recommendation: {recommendation}"
        )

        return {"output": final_answer}


def build_agent():
    return FraudAgent()


def run_agent(agent, query: str) -> str:
    try:
        result = agent.invoke({"input": query})
        return result.get("output", str(result))
    except Exception as e:
        return f"Agent error: {str(e)}"


if __name__ == "__main__":
    agent = build_agent()
    answer = run_agent(agent, "Is a $3000 transaction at 3AM suspicious?")
    print(answer)