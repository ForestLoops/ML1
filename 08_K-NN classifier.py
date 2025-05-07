import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score

# Generate synthetic binary classification dataset
X, y = make_classification(n_samples=300, n_features=6, n_classes=2, n_informative=5, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize K-NN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# === Output Metrics ===
print("=== K-NN Classifier (Train-Test Split Evaluation) ===")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Error Rate   : {error_rate:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"AUC Score    : {auc:.4f}")

# === K-Fold Cross-Validation ===
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')
recall_scores = cross_val_score(knn, X, y, cv=kfold, scoring='recall')
f1_scores = cross_val_score(knn, X, y, cv=kfold, scoring='f1')
auc_scores = cross_val_score(knn, X, y, cv=kfold, scoring='roc_auc')

print("\n=== 5-Fold Cross-Validation ===")
print(f"Avg Accuracy : {acc_scores.mean():.4f}")
print(f"Avg Recall   : {recall_scores.mean():.4f}")
print(f"Avg F1 Score : {f1_scores.mean():.4f}")
print(f"Avg AUC      : {auc_scores.mean():.4f}")
