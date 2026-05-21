import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

X = diabetes_X[:, np.newaxis, 2]
y = diabetes_y

model = linear_model.LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

print("Coefficient:", model.coef_)

plt.scatter(X, y)
plt.plot(X, y_pred)

plt.title("Linear Regression")

plt.show()