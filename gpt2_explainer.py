from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = None
gpt2_model = None

def load_gpt2():
    """Load GPT-2 only once when needed."""
    global tokenizer, gpt2_model
    if tokenizer is None or gpt2_model is None:
        print("Loading GPT-2 model...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_model.eval()

def build_reason(fraud_prob, amount, time_hour, location_flag):
    """
    Create a basic model-based explanation.
    GPT-2 will only rewrite this reason, not invent a reason.
    """
    reasons = []
    if amount > 500:
        reasons.append("transaction amount exceeds typical fraud range")
    if time_hour < 6:
        reasons.append("late-night transaction time")
    if location_flag == 1:
        reasons.append("transaction from a low population area")
    if fraud_prob >= 0.70:
        risk = "critical"
    elif fraud_prob >= 0.30:
        risk = "medium"
    else:
        risk = "low"
    if not reasons:
        reasons.append("normal transaction pattern")
    reason_text = (
        f"The fraud risk is {risk}. "
        f"The model predicted a fraud probability of {fraud_prob * 100:.1f}%. "
        f"The main reason is: {', '.join(reasons)}."
    )
    return reason_text

def generate_fraud_explanation(fraud_prob, amount, time_hour, location_flag):
    """
    Generate a short fraud explanation using GPT-2.
    If GPT-2 fails or gives poor output, return the safe rule-based explanation.
    """
    base_reason = build_reason(
        fraud_prob=fraud_prob,
        amount=amount,
        time_hour=time_hour,
        location_flag=location_flag
    )
    try:
        load_gpt2()
        prompt = (
            "Rewrite this fraud detection reason in simple English. "
            "Do not add extra facts.\n\n"
            f"Reason: {base_reason}\n\n"
            "Explanation:"
        )
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = gpt2_model.generate(
                inputs,
                max_new_tokens=45,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = full_text.split("Explanation:")[-1].strip()

        if len(explanation) < 20:
            return base_reason
        return explanation
    except Exception:
        return base_reason
        
if __name__ == "__main__":
    result = generate_fraud_explanation(
        fraud_prob=0.96,
        amount=995.57,
        time_hour=2,
        location_flag=1
    )
    print(result)