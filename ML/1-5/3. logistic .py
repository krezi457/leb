import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load the built-in Breast Cancer dataset (Binary Classification)
cancer = datasets.load_breast_cancer()

# Use only one feature ('mean radius', the 1st column) to easily visualize the curve
X = cancer.data[:, np.newaxis, 0]
y = cancer.target  # 0: Malignant, 1: Benign

# 2. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# Print model performance metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Plot outputs (The Sigmoid Curve)
# Generate a smooth range of X values to draw the curve
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# predict_proba returns probabilities for class 0 and class 1. We grab class 1.
y_prob = model.predict_proba(X_range)[:, 1]

plt.figure(figsize=(8, 6))

# Plot the actual test data (will be strictly at y=0 or y=1)
plt.scatter(X_test, y_test, color="black", alpha=0.5, label="Actual Data (Test Set)")

# Plot the logistic S-curve
plt.plot(X_range, y_prob, color="red", linewidth=3, label="Logistic Probability Curve")

# Find and plot the decision boundary (where probability crosses 50%)
# The model classifies everything left of this line as one class, and right as the other
boundary_idx = np.argmin(np.abs(y_prob - 0.5))
boundary_x = X_range[boundary_idx][0]
plt.axvline(x=boundary_x, color='blue', linestyle='--', label=f'Decision Boundary (Radius ~{boundary_x:.1f})')

plt.xlabel("Tumor Mean Radius")
plt.ylabel("Probability of Being Benign (Class 1)")
plt.title("Logistic Regression: Tumor Classification")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

# Display the graph
plt.show()