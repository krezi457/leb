import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)

print(model.predict(X))

plt.scatter(X[:,0], X[:,1], c=y)

plt.title("KNN")

plt.show()