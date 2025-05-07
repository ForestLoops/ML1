import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Generate a non-linear dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze()**2 + 3 * X.squeeze() + 5 + np.random.randn(100) * 10  # Quadratic relationship

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a polynomial regression pipeline (degree 2)
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Train-Test Split Evaluation ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score               : {r2:.4f}")

# === K-Fold Cross-Validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)

print("\n=== 5-Fold Cross-Validation ===")
print(f"Average MSE : {mse_scores.mean():.4f}")
print(f"Average R²  : {r2_scores.mean():.4f}")
