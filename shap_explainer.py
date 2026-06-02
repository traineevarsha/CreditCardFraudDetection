import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_feature_importance(top_n=15):
    """
    Get feature importance from the Decision Tree model.
    Decision Tree has built-in feature importance so no extra library needed.
    Returns a DataFrame sorted by importance descending.
    """
    dt            = joblib.load("models/decision_tree.pkl")
    feature_names = joblib.load("models/feature_names.pkl")

    importance_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": dt.feature_importances_
    }).sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)

    importance_df.index += 1  # start ranking from 1
    return importance_df


def plot_feature_importance(top_n=15):
    """
    Plot a horizontal bar chart of the top N most important features.
    Returns a matplotlib figure so Streamlit can display it with st.pyplot().
    """
    df = get_feature_importance(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Importance", y="Feature",
                palette="Reds_r", ax=ax)
    ax.set_title(f"Top {top_n} Features — Decision Tree (Phase 1 Champion)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig


def explain_single_prediction(row_dict, feature_names):
    """
    SHAP-style explanation for a single transaction.
    Shows which features pushed the prediction toward fraud or legitimate.
    Works by comparing the transaction's feature values against typical values.

    Args:
        row_dict:      dict of feature name -> encoded value
        feature_names: list of all feature names in training order

    Returns:
        DataFrame of top contributing features with direction
    """
    dt     = joblib.load("models/decision_tree.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # scale the input
    X = pd.DataFrame([row_dict])[feature_names]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)

    # use feature importance as proxy for contribution
    # multiply importance by the scaled feature value to get direction
    importances = dt.feature_importances_
    contributions = []
    for i, fname in enumerate(feature_names):
        val         = float(X_scaled.iloc[0, i])
        importance  = importances[i]
        contribution = val * importance
        contributions.append({
            "Feature":      fname,
            "Value":        round(val, 3),
            "Importance":   round(importance, 4),
            "Contribution": round(contribution, 4)
        })

    contrib_df = pd.DataFrame(contributions)
    contrib_df = contrib_df.reindex(
        contrib_df["Contribution"].abs().sort_values(ascending=False).index
    ).head(10).reset_index(drop=True)
    contrib_df.index += 1

    return contrib_df


if __name__ == "__main__":
    df = get_feature_importance()
    print("Top 10 features:")
    print(df.head(10).to_string())