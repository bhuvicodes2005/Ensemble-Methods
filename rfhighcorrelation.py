import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------- #
# Generate High-Correlation Dataset
# ------------------------------------------- #
def generate_high_correlation_data(n_samples=1000, n_informative=5, total_features=20, random_state=42):
    X_base, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        flip_y=0.01,
        class_sep=1.0,
        random_state=random_state
    )
    np.random.seed(random_state)
    correlated_features = []

    for i in range(total_features - n_informative):
        base_idx = np.random.randint(0, n_informative)
        noise = np.random.normal(0, 0.05, size=n_samples)
        correlated_feature = X_base[:, base_idx] + noise
        correlated_features.append(correlated_feature.reshape(-1, 1))

    X_correlated = np.hstack([X_base] + correlated_features)
    return X_correlated, y

X, y = generate_high_correlation_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------- #
# Visualize Feature Correlation Matrix
# ------------------------------------------- #
plt.figure(figsize=(12, 8))
corr_matrix = pd.DataFrame(X).corr()
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# ------------------------------------------- #
# Strength, Correlation, c/s² Calculator
# ------------------------------------------- #
def forest_strength_and_correlation(rf_model, X, y):
    n_trees = len(rf_model.estimators_)
    n_samples = X.shape[0]
    classes = rf_model.classes_
    n_classes = len(classes)

    tree_preds = np.array([tree.predict(X) for tree in rf_model.estimators_])
    margins = []
    raw_margins_matrix = []

    for i in range(n_samples):
        y_true = y[i]
        votes = np.bincount(tree_preds[:, i].astype(int), minlength=n_classes)
        true_class_votes = votes[y_true]
        max_other_votes = np.max(np.delete(votes, y_true))
        margin = (true_class_votes - max_other_votes) / n_trees
        margins.append(margin)
        row = [1 if pred == y_true else -1 for pred in tree_preds[:, i]]
        raw_margins_matrix.append(row)

    margins = np.array(margins)
    strength = np.mean(margins)
    raw_margins_matrix = np.array(raw_margins_matrix).T

    correlations = []
    for i, j in itertools.combinations(range(n_trees), 2):
        corr, _ = pearsonr(raw_margins_matrix[i], raw_margins_matrix[j])
        correlations.append(corr)

    avg_corr = np.mean(correlations)
    cs_ratio = avg_corr / (strength ** 2) if strength != 0 else np.inf
    return strength, avg_corr, cs_ratio

# ------------------------------------------- #
# Train RF with given max_features
# ------------------------------------------- #
def train_rf(X_train, X_test, y_train, y_test, max_features, label=""):
    rf = RandomForestClassifier(n_estimators=100, max_features=max_features, oob_score=True, bootstrap=True, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    strength, corr, cs_ratio = forest_strength_and_correlation(rf, X_test, y_test)
    print(f"🔹 {label} | Accuracy: {acc:.4f} | OOB: {rf.oob_score_:.4f} | Strength: {strength:.4f} | Corr: {corr:.4f} | c/s²: {cs_ratio:.4f}")
    return {
        "Method": label,
        "Accuracy": acc,
        "OOB": rf.oob_score_,
        "Strength": strength,
        "Correlation": corr,
        "c/s²": cs_ratio
    }

# ------------------------------------------- #
# Run Experiments
# ------------------------------------------- #
results = []
M = X.shape[1]

results.append(train_rf(X_train, X_test, y_train, y_test, max_features=1, label="RI (F=1)"))
results.append(train_rf(X_train, X_test, y_train, y_test, max_features=int(np.log2(M)) + 1, label="RI (F=log2(M)+1)"))

# AdaBoost
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
acc_ada = accuracy_score(y_test, y_pred_ada)
print(f"🔸 AdaBoost | Accuracy: {acc_ada:.4f}")
results.append({
    "Method": "AdaBoost",
    "Accuracy": acc_ada,
    "OOB": np.nan,
    "Strength": np.nan,
    "Correlation": np.nan,
    "c/s²": np.nan
})

# ------------------------------------------- #
# Results & Plot
# ------------------------------------------- #
results_df = pd.DataFrame(results)
print("\n📊 Final Results:")
print(results_df[["Method", "Accuracy", "OOB", "Strength", "Correlation", "c/s²"]].round(4))

plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Method", y="Accuracy")
plt.title("Accuracy Comparison (RI F=1, F=log2(M)+1, AdaBoost)")
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.show()
