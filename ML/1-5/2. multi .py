import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load the in-built diabetes dataset
# We keep all 10 features this time
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 2. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    diabetes_X, diabetes_y, test_size=0.2, random_state=42
)

# 3. Create and train the multiple linear regression object
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions using the testing set
y_pred = model.predict(X_test)

# Print model performance metrics
print("Coefficients for each of the 10 features:")
for i, coef in enumerate(model.coef_):
    print(f"Feature {i+1}: {coef:.2f}")

print(f"\nMean squared error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination (R²): {r2_score(y_test, y_pred):.2f}")

# 5. Plot outputs (Actual vs. Predicted)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="black", alpha=0.6, label="Predictions")

# Draw the 45-degree reference line (Perfect Prediction Line)
# If the model were perfect, every point would lie exactly on this line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color="blue", linewidth=3, linestyle="--", label="Perfect Prediction")

plt.xlabel("Actual Disease Progression")
plt.ylabel("Predicted Disease Progression")
plt.title("Multiple Linear Regression: Actual vs. Predicted")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

# Display the graph
plt.show()