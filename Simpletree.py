from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# Train a simple tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10)
dt.fit(X_train, y_train)
preds = dt.predict(X_test)
acc = accuracy_score(y_test, preds)

precision = precision_score(y_test, preds, average='weighted')
recall = recall_score(y_test, preds, average='weighted')
f1 = f1_score(y_test, preds, average='weighted')
cm = confusion_matrix(y_test, preds)
error_rate = 1 - acc

print("Decision Tree Accuracy:", acc)
print("Decision Tree Precision:", precision)
print("Decision Tree Recall:", recall)
print("Decision Tree F1 Score:", f1)
print("Decision Tree Error Rate:", error_rate)
print("Confusion Matrix:\n", cm)
