
# Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# 1. Load diabetes dataset
data = load_diabetes()
X = data.data
y = data.target
features = data.feature_names

# 2. Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create models
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000)
}

results = []

# 3. Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Count zero coefficients
    zero_count = np.sum(model.coef_ == 0)

    results.append([name, round(rmse, 2), zero_count])

    print("\n", name, "Regression")
    print("Test RMSE:", round(rmse, 2))

    # Print 5 largest coefficients for Linear Regression
    if name == "Linear":
        coef_table = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        })
        coef_table["Absolute"] = abs(coef_table["Coefficient"])

        print("\n5 largest coefficients:")
        print(
            coef_table.sort_values(
                "Absolute", ascending=False
            )[["Feature", "Coefficient"]].head(5)
        )

    # Print coefficients for all models
    print("\nCoefficients:")
    for feature, coef in zip(features, model.coef_):
        print(feature, ":", round(coef, 3))

# 4. Print comparison table
print("\nModel Comparison")
print("Model | Test RMSE | Number of zero coefficients")

for row in results:
    print(f"{row[0]:7} | {row[1]:9.2f} | {row[2]}")

# 5. Count zero coefficients in Lasso
lasso = models["Lasso"]
print("\nLasso zero coefficients:", np.sum(lasso.coef_ == 0))
