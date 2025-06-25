import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.utils import resample
import time
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Load and downsample MNIST
# -------------------------------
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(np.uint8)
X_train, X_test, y_train, y_test = train_test_split(X[:10000], y[:10000], test_size=0.3, random_state=42)

# -------------------------------
# Forest-RI as per Breiman (random features per node)
# -------------------------------
class ForestRI(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        
        self.model = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_features=self.max_features,
                max_depth=None,
                splitter="best"
            ),
            n_estimators=self.n_estimators,
            bootstrap=True,
            n_jobs=-1,
            random_state=self.random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# -------------------------------
# Forest-RC as per Breiman (random linear combos per tree)
# -------------------------------
class ForestRC(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, L=3, F=5, random_state=42):
        self.n_estimators = n_estimators
        self.L = L
        self.F = F
        self.random_state = random_state
        self.trees = []
        self.feature_maps = []

    def _generate_combos(self, X, rng):
        combos = []
        for _ in range(self.F):
            idx = rng.choice(X.shape[1], self.L, replace=False)
            weights = rng.uniform(-1, 1, self.L)
            combos.append((idx, weights))
        return combos

    def _transform(self, X, combos):
        features = []
        for idx, weights in combos:
            features.append(np.dot(X[:, idx], weights))
        return np.array(features).T

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.trees = []
        self.feature_maps = []

        for i in range(self.n_estimators):
            Xs, ys = resample(X, y, replace=True, random_state=rng.randint(10000))
            combos = self._generate_combos(Xs, rng)
            X_new = self._transform(Xs, combos)
            tree = DecisionTreeClassifier(max_depth=None, random_state=rng.randint(10000))
            tree.fit(X_new, ys)
            self.trees.append(tree)
            self.feature_maps.append(combos)
        return self

    def predict(self, X):
        preds = []
        for tree, combos in zip(self.trees, self.feature_maps):
            X_new = self._transform(X, combos)
            preds.append(tree.predict(X_new))
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=np.array(preds))

# -------------------------------
# Define all models including AdaBoost
# -------------------------------
print("Training models...")

models = {
    "Forest-RI (F=1)": ForestRI(n_estimators=100, max_features=1),
    "Forest-RI (F=10)": ForestRI(n_estimators=100, max_features=10),
    "Forest-RC": ForestRC(n_estimators=100, L=3, F=5),
    "AdaBoost (depth=2)": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=50,
        algorithm="SAMME",  # Changed SAMME.R to SAMME
        random_state=42
    )
}

# Initialize the results dictionary
results = {}

# Run models 5 times and average the results
for name, model in models.items():
    print(f"Training {name}...")
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    training_times = []
    confusion_matrices = []  # To store confusion matrices

    for run in range(5):  # Run each model 5 times
        start_time = time.time()  # Start time tracking
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        end_time = time.time()  # End time tracking
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        class_report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)
        
        # Collect metrics for averaging
        accuracies.append(acc)
        precisions.append(class_report['accuracy'])
        recalls.append(class_report['macro avg']['recall'])
        f1_scores.append(class_report['macro avg']['f1-score'])
        training_times.append(end_time - start_time)  # Store the training time
        confusion_matrices.append(cm)  # Store confusion matrices
    
    # Store average results and confusion matrix
    results[name] = {
        "average_accuracy": np.mean(accuracies),
        "average_precision": np.mean(precisions),
        "average_recall": np.mean(recalls),
        "average_f1_score": np.mean(f1_scores),
        "average_training_time": np.mean(training_times),
        "confusion_matrices": confusion_matrices  # Store all confusion matrices for this model
    }

# -------------------------------
# Results Summary
# -------------------------------
print("\nModel Accuracy Comparison:")
for name, res in results.items():
    print(f"{name:20s}: {res['average_accuracy']*100:.2f}%")
    print(f"Average Training Time: {res['average_training_time']:.4f} seconds")
    print(f"Average Precision: {res['average_precision']:.4f}")
    print(f"Average Recall: {res['average_recall']:.4f}")
    print(f"Average F1 Score: {res['average_f1_score']:.4f}")
    print("-" * 50)

# -------------------------------
# Plot confusion matrix (example: Forest-RC)
# -------------------------------
def plot_conf_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot confusion matrix for each model using the average confusion matrix
for name, res in results.items():
    avg_conf_matrix = np.mean(res["confusion_matrices"], axis=0).astype(int)  # Average the confusion matrices
    plot_conf_matrix(avg_conf_matrix, f"{name} Averaged Confusion Matrix")
