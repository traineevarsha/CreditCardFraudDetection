from langchain.tools import Tool
import joblib
import pandas as pd
import re

from gpt2_explainer import generate_fraud_explanation
from nltk_processor import process_explanation

FRAUD_THRESHOLD = 0.30
HIGH_RISK_CATEGORIES = ["shopping_net", "misc_net", "grocery_pos"]

def extract_amount_hour_category(query):
    query = query.lower()

    # Extract amount after $
    amount_match = re.search(r"\$?\s*(\d+\.?\d*)", query)
    amount = float(amount_match.group(1)) if amount_match else 500.0

    # Extract hour 
    time_match = re.search(r"(\d{1,2})\s*(am|pm)", query)

    if time_match:
        hour = int(time_match.group(1))
        period = time_match.group(2)

        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
    else:
        numbers = re.findall(r"\d+\.?\d*", query)
        hour = float(numbers[1]) if len(numbers) > 1 else 12.0

    if hour > 23:
        hour = 12.0

    category = ""
    for cat in HIGH_RISK_CATEGORIES:
        if cat in query:
            category = cat
            break

    return amount, hour, category

def predict_transaction(amount, hour, category):
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    encoders = joblib.load("models/encoders.pkl")

    row = {name: 0 for name in feature_names}

    row["amt"] = amount
    row["hour"] = hour

    if "is_night" in feature_names:
        row["is_night"] = 1 if hour < 6 else 0
    if "is_high_amount" in feature_names:
        row["is_high_amount"] = 1 if amount > 500 else 0
    if "category" in feature_names and category in encoders["category"].classes_:
        row["category"] = int(encoders["category"].transform([category])[0])

    X = pd.DataFrame([row])[feature_names]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)
    prob = model.predict_proba(X_scaled)[0][1]

    # Simple risk adjustment, same idea as main app
    rule_score = 0
    if amount > 1000:
        rule_score += 1
    if hour < 6:
        rule_score += 1
    if category in HIGH_RISK_CATEGORIES:
        rule_score += 1
    prob = min(prob + (rule_score * 0.10), 1.0)
    verdict = "FRAUD" if prob >= FRAUD_THRESHOLD else "LEGITIMATE"
    return verdict, prob

def fraud_tool(query):
    amount, hour, category = extract_amount_hour_category(query)
    verdict, prob = predict_transaction(amount, hour, category)
    explanation = generate_fraud_explanation(
        prob,
        amount,
        int(hour),
        0
    )
    processed = process_explanation(explanation)
    keywords = processed["keywords"]
    clean_explanation = processed["clean"].replace(
        "The fraud risk is not a problem for the model.",
        "The fraud risk is medium."
    )

    if prob >= 0.70:
        action = "BLOCK this transaction immediately."
    elif prob >= FRAUD_THRESHOLD:
        action = "FLAG this transaction for manual review."
    else:
        action = "APPROVE this transaction."

    return (
        f"Prediction: {verdict}\n\n"
        f"Fraud Probability: {prob * 100:.1f}%\n\n"
        f"GPT-2 Explanation: {clean_explanation}\n\n"
        f"NLTK Keywords: {', '.join(keywords) if keywords else 'None'}\n\n"
        f"Recommended Action: {action}"
    )

tools = [
    Tool(
        name="FraudAssistant",
        func=fraud_tool,
        description="Analyzes a transaction and returns fraud prediction, explanation, keywords, and action."
    )
]

class FraudAgent:
    def invoke(self, inputs):
        query = inputs.get("input", "")
        return {"output": fraud_tool(query)}

def build_agent():
    return FraudAgent()

def run_agent(agent, query):
    try:
        result = agent.invoke({"input": query})
        return result["output"]
    except Exception as e:
        return f"Agent error: {str(e)}"

if __name__ == "__main__":
    agent = build_agent()
    print(run_agent(agent, "Is a $3000 shopping_net transaction at 3AM suspicious?"))