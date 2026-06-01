def rule_engine(amount, category, hour):

    score = 0
    reasons = []

    if amount > 1200:
        score += 1
        reasons.append("High transaction amount")

    if hour < 6:
        score += 1
        reasons.append("Late night transaction")

    if category in ["shopping_net", "misc_net", "grocery_pos"]:
        score += 1
        reasons.append("High-risk merchant category")

    return score, reasons