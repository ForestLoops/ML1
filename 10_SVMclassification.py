import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score

# Generate synthetic classification data
X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_redundant=1, random_state=42)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVM classifier (using RBF kernel)
model = SVC(kernel='rbf', probability=True, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# Output metrics
print("=== SVM - Train-Test Evaluation ===")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Error Rate   : {error_rate:.4f}")
print(f"True Positives  (TP): {tp}")
print(f"True Negatives  (TN): {tn}")
print(f"False Positives(FP): {fp}")
print(f"False Negatives(FN): {fn}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"AUC Score    : {auc:.4f}")

# === K-Fold Cross-Validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

acc_cv = cross_val_score(model, X, y, scoring='accuracy', cv=kf)
recall_cv = cross_val_score(model, X, y, scoring='recall', cv=kf)
f1_cv = cross_val_score(model, X, y, scoring='f1', cv=kf)
auc_cv = cross_val_score(model, X, y, scoring='roc_auc', cv=kf)

print("\n=== 5-Fold Cross-Validation ===")
print(f"Avg Accuracy : {acc_cv.mean():.4f}")
print(f"Avg Recall   : {recall_cv.mean():.4f}")
print(f"Avg F1 Score : {f1_cv.mean():.4f}")
print(f"Avg AUC      : {auc_cv.mean():.4f}")
