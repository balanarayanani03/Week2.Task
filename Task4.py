
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =====================================
# PART 1: Binary Logistic Regression
# =====================================

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict classes
y_pred = model.predict(X_test)

print("Binary Logistic Regression")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

# Get probabilities for first 5 test rows
probs = model.predict_proba(X_test[:5])
preds = model.predict(X_test[:5])

print("\nProbabilities for first 5 rows:")
print(pd.DataFrame(
    probs,
    columns=["Class 0 probability", "Class 1 probability"]
).round(3))

print("\nPredicted classes:", preds)
print("Actual classes:", y_test[:5])

# =====================================
# PART 2: Change threshold from 0.5 to 0.3
# =====================================

# Probability of class 1
prob_class1 = model.predict_proba(X_test)[:, 1]

pred_05 = (prob_class1 >= 0.5).astype(int)
pred_03 = (prob_class1 >= 0.3).astype(int)

print("\nThreshold Comparison (first 10 rows)")
print(pd.DataFrame({
    "Probability": prob_class1[:10].round(3),
    "Prediction at 0.5": pred_05[:10],
    "Prediction at 0.3": pred_03[:10]
}))

print("\nPositive predictions at threshold 0.5:",
      np.sum(pred_05 == 1))
print("Positive predictions at threshold 0.3:",
      np.sum(pred_03 == 1))

# =====================================
# PART 3: Bonus - Threshold plot
# =====================================

thresholds = np.arange(0.1, 1.0, 0.1)
positive_counts = []

for threshold in thresholds:
    predictions = (prob_class1 >= threshold).astype(int)
    positive_counts.append(np.sum(predictions == 1))

plt.plot(thresholds, positive_counts, marker="o")
plt.xlabel("Decision Threshold")
plt.ylabel("Number of Positive Predictions")
plt.title("Threshold vs Positive Predictions")
plt.xticks(thresholds)
plt.grid(True)
plt.show()

# =====================================
# PART 4: Multi-class Logistic Regression
# =====================================

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2,
    random_state=42, stratify=y_iris
)

iris_model = LogisticRegression(max_iter=1000)
iris_model.fit(X_train_i, y_train_i)

# Get probabilities for 3 flowers
flower_probs = iris_model.predict_proba(X_test_i[:3])

print("\nIris Probability Table")
print(pd.DataFrame(
    flower_probs,
    columns=iris.target_names
).round(3))

print("\nPredicted flower classes:",
      iris.target_names[iris_model.predict(X_test_i[:3])])
print("Actual flower classes:",
      iris.target_names[y_test_i[:3]])
