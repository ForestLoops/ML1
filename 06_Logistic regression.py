import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve

# Generate synthetic classification data
X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=0, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
recall = recall_score(y_test, y_pred)                   # Sensitivity
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# === Output Metrics ===
print("=== Logistic Regression (Train-Test Split) ===")
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
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracy_cv = cross_val_score(model, X, y, scoring='accuracy', cv=kf)
recall_cv = cross_val_score(model, X, y, scoring='recall', cv=kf)
f1_cv = cross_val_score(model, X, y, scoring='f1', cv=kf)
auc_cv = cross_val_score(model, X, y, scoring='roc_auc', cv=kf)

print(f"\n=== {k}-Fold Cross-Validation ===")
print(f"Average Accuracy : {accuracy_cv.mean():.4f}")
print(f"Average Recall   : {recall_cv.mean():.4f}")
print(f"Average F1 Score : {f1_cv.mean():.4f}")
print(f"Average AUC      : {auc_cv.mean():.4f}")
