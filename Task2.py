
# Import libraries
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 1. Five-fold cross-validation (correct way)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print("CV scores:", np.round(scores, 2))
print("Mean:", round(scores.mean(), 2))
print("Standard deviation:", round(scores.std(), 2))

# 2. Demonstrate data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# WRONG: Scale the whole dataset before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_leak = X_scaled[:len(X_train)]
# Use the actual training indices for a fair comparison
# (split indices are created again below)

indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

X_train_leak = X_scaled[train_idx]
X_test_leak = X_scaled[test_idx]

wrong_model = LogisticRegression(max_iter=5000)
wrong_model.fit(X_train_leak, y[train_idx])
wrong_score = accuracy_score(y[test_idx], wrong_model.predict(X_test_leak))

# CORRECT: Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X[train_idx])
X_test_scaled = scaler.transform(X[test_idx])

correct_model = LogisticRegression(max_iter=5000)
correct_model.fit(X_train_scaled, y[train_idx])
correct_score = accuracy_score(
    y[test_idx], correct_model.predict(X_test_scaled)
)

print("\nData Leakage Comparison")
print("Wrong way (leakage):", round(wrong_score, 4))
print("Correct way:", round(correct_score, 4))
