import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

model = SVC()

model.fit(X, y)

print(model.predict(X))

plt.scatter(X[:,0], X[:,1], c=y)

plt.title("SVM")

plt.show()