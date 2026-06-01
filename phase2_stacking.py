# phase2.py
# Phase 2: Train a Stacking Ensemble using the 3 models from Phase 1.
# Run this AFTER phase1.py
#
# Owner: [Team Member 3]

import joblib
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score, roc_auc_score

# ------------------------------------------------------------------
# STEP 1: Load models and data saved by phase1.py
# ------------------------------------------------------------------
print("Loading Phase 1 models...")
dt  = joblib.load("models/decision_tree.pkl")
lr  = joblib.load("models/logistic_regression.pkl")
mlp = joblib.load("models/neural_network.pkl")

X_train, y_train = joblib.load("models/train_data.pkl")
X_test,  y_test  = joblib.load("models/test_data.pkl")

phase1_best_name   = open("models/best_model_name.txt").read().strip()
phase1_best_recall = joblib.load("models/best_model.pkl")
# get recall of best phase 1 model to compare against
phase1_recall = recall_score(y_test, phase1_best_recall.predict(X_test))

print(f"Phase 1 champion: {phase1_best_name} (Recall = {phase1_recall:.3f})")

# ------------------------------------------------------------------
# STEP 2: Build and train the Stacking Ensemble
# stacking = base models each make a prediction, then a meta model
# learns which model to trust for the final decision
# ------------------------------------------------------------------
print("\nPhase 2 — Training Stacking Ensemble...")
stacking = StackingClassifier(
    estimators=[("dt", dt), ("lr", lr), ("mlp", mlp)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,      # 3-fold cross validation prevents data leakage
    n_jobs=-1  # use all CPU cores to speed up training
)
stacking.fit(X_train, y_train)

# ------------------------------------------------------------------
# STEP 3: Evaluate the stacking model
# ------------------------------------------------------------------
stack_pred   = stacking.predict(X_test)
stack_prob   = stacking.predict_proba(X_test)[:, 1]
stack_report = classification_report(y_test, stack_pred, output_dict=True)
stack_recall = stack_report["1"]["recall"]
stack_f1     = stack_report["1"]["f1-score"]
stack_auc    = roc_auc_score(y_test, stack_prob)

print(classification_report(y_test, stack_pred, target_names=["Legit", "Fraud"]))
print(f"Stacking Recall:  {stack_recall:.3f}")
print(f"Stacking F1:      {stack_f1:.3f}")
print(f"Stacking ROC-AUC: {stack_auc:.3f}")

# ------------------------------------------------------------------
# STEP 4: Compare Phase 1 vs Phase 2 and save the best overall model
# ------------------------------------------------------------------
print(f"\nPhase 1 best Recall: {phase1_recall:.3f}")
print(f"Phase 2 Recall:      {stack_recall:.3f}")

if stack_recall > phase1_recall:
    print("✅ Stacking improved on Phase 1!")
    joblib.dump(stacking, "models/best_model.pkl")
    open("models/best_model_name.txt", "w").write("Stacking Ensemble")
else:
    print(f"ℹ️  Phase 1 champion ({phase1_best_name}) still wins.")
    print("   Phase 1 best model remains saved as best_model.pkl")

joblib.dump(stacking, "models/stacking_model.pkl")
print("\nPhase 2 complete.")
print("Now run:  streamlit run app.py")