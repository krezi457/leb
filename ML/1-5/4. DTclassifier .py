import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load the built-in Iris dataset (Multiclass Classification)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train the Decision Tree Classifier
# We limit max_depth to 3 to prevent overfitting and keep the graph readable
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# Print model performance metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Plot the decision tree structure
plt.figure(figsize=(14, 10))
plot_tree(
    model, 
    feature_names=iris.feature_names,  # Names of the columns (e.g., 'petal length')
    class_names=iris.target_names,     # Names of the species (setosa, versicolor, virginica)
    filled=True,                       # Colors the nodes based on the majority class
    rounded=True,                      # Rounds the corners of the node boxes
    fontsize=10
)

plt.title("Decision Tree Classifier: Iris Species Logic", fontsize=14)

# Display the graph
plt.show()