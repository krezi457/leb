import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature (BMI)
X = diabetes_X[:, np.newaxis, 2]
y = diabetes_y

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = DecisionTreeRegressor(max_depth=3)

# Train model
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print results
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Create smooth graph
X_grid = np.arange(X.min(), X.max(), 0.001).reshape(-1,1)

# Predict graph values
y_grid_pred = model.predict(X_grid)

# Plot graph
plt.scatter(X_test, y_test, color="black")

plt.plot(X_grid, y_grid_pred, color="red")

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Decision Tree Regression")

plt.show()