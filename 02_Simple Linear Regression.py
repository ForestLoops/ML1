import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# Generate a simple regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("=== Train-Test Split Evaluation ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score               : {r2:.4f}")

# === K-Fold Cross-Validation ===
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)

print(f"\n=== {k}-Fold Cross-Validation ===")
print(f"Average MSE : {mse_scores.mean():.4f}")
print(f"Average R²  : {r2_scores.mean():.4f}")
