from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data[:, :2]

model = KMeans(n_clusters=3)

model.fit(X)

y_kmeans = model.predict(X)

print("Cluster Labels:")
print(y_kmeans)

plt.scatter(X[:,0], X[:,1], c=y_kmeans)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")

plt.show()