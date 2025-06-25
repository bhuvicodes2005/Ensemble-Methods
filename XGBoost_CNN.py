from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import numpy as np
from tensorflow.keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# Run XGBoost 5 times
xgb_accs, xgb_times = [], []
xgb_precisions, xgb_recalls, xgb_f1s, xgb_errors = [], [], [], []

for _ in range(5):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    xgb_accs.append(acc)
    xgb_times.append(end - start)

    precision = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    error_rate = 1 - acc

    xgb_precisions.append(precision)
    xgb_recalls.append(recall)
    xgb_f1s.append(f1)
    xgb_errors.append(error_rate)

print("XGBoost Avg Accuracy:", np.mean(xgb_accs))
print("XGBoost Avg Precision:", np.mean(xgb_precisions))
print("XGBoost Avg Recall:", np.mean(xgb_recalls))
print("XGBoost Avg F1 Score:", np.mean(xgb_f1s))
print("XGBoost Avg Error Rate:", np.mean(xgb_errors))
print("XGBoost Avg Time (s):", np.mean(xgb_times))

# Display Confusion Matrix for the last run
print("Confusion Matrix (last run):\n", confusion_matrix(y_test, preds))
