import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generate a dataset
X, y = make_regression(n_samples=150, n_features=10, noise=20, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models with regularization strength (alpha)
lasso_model = Lasso(alpha=1.0)
ridge_model = Ridge(alpha=1.0)

# Train the models
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# Predictions
lasso_pred = lasso_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)

# === LASSO ===
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("=== Lasso Regression (Train-Test Split) ===")
print(f"MSE       : {lasso_mse:.4f}")
print(f"R² Score  : {lasso_r2:.4f}")

# === RIDGE ===
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("\n=== Ridge Regression (Train-Test Split) ===")
print(f"MSE       : {ridge_mse:.4f}")
print(f"R² Score  : {ridge_r2:.4f}")

# === K-Fold Cross-Validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lasso CV
lasso_cv_mse = -cross_val_score(Lasso(alpha=1.0), X, y, scoring='neg_mean_squared_error', cv=kf)
lasso_cv_r2 = cross_val_score(Lasso(alpha=1.0), X, y, scoring='r2', cv=kf)

# Ridge CV
ridge_cv_mse = -cross_val_score(Ridge(alpha=1.0), X, y, scoring='neg_mean_squared_error', cv=kf)
ridge_cv_r2 = cross_val_score(Ridge(alpha=1.0), X, y, scoring='r2', cv=kf)

print("\n=== Lasso Regression (5-Fold CV) ===")
print(f"Average MSE : {lasso_cv_mse.mean():.4f}")
print(f"Average R²  : {lasso_cv_r2.mean():.4f}")

print("\n=== Ridge Regression (5-Fold CV) ===")
print(f"Average MSE : {ridge_cv_mse.mean():.4f}")
print(f"Average R²  : {ridge_cv_r2.mean():.4f}")
