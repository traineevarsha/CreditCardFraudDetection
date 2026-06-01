import joblib
import numpy as np

model = joblib.load("models/best_model.pkl")
X_test, y_test = joblib.load("models/test_data.pkl")

probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

print("Best model:", open("models/best_model_name.txt").read().strip())
print("Max fraud probability:", probs.max())
print("Mean fraud probability:", probs.mean())
print("Min fraud probability:", probs.min())

print("Predicted fraud count:", np.sum(preds == 1))
print("Actual fraud count:", np.sum(y_test == 1))

print("Above 0.3:", np.sum(probs >= 0.3))
print("Above 0.5:", np.sum(probs >= 0.5))
print("Above 0.7:", np.sum(probs >= 0.7))