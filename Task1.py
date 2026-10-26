
# 1. Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 2. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Class balance:")
print(data.target_names)
print(__import__("numpy").bincount(y))

# 3. Split data: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# 4. Train decision trees with different depths
depths = [1, 3, 5, 10, None]
train_acc = []
val_acc = []

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth, random_state=42
    )
    model.fit(X_train, y_train)

    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    val_acc.append(accuracy_score(y_val, model.predict(X_val)))

# 5. Print accuracy table
print("\nDepth | Train Accuracy | Validation Accuracy")
for d, tr, va in zip(depths, train_acc, val_acc):
    print(f"{str(d):5} | {tr:.4f}         | {va:.4f}")

# 6. Plot depth vs accuracy
labels = [str(d) for d in depths]

plt.figure(figsize=(8, 5))
plt.plot(labels, train_acc, marker="o", label="Train Accuracy")
plt.plot(labels, val_acc, marker="o", label="Validation Accuracy")

# Mark possible overfitting
best_index = val_acc.index(max(val_acc))
plt.axvline(
    x=best_index, color="red", linestyle="--",
    label="Best validation accuracy"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Overfitting Experiment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
