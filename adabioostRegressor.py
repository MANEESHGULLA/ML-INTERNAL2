# adaboost regressor
# --- Step 1: Import Libraries ---
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 2: Generate Synthetic Regression Data ---
X, y = make_regression(n_samples=1000, n_features=10,
                       noise=15, random_state=0)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("No. of training samples:", X_train.shape)
print("No. of testing samples:", X_test.shape)

# --- Step 3: Create Base Learner ---
base_learner = DecisionTreeRegressor(max_depth=3)

# --- Step 4: Create AdaBoost Regressor Model ---
model = AdaBoostRegressor(estimator=base_learner,
                          n_estimators=100,
                          learning_rate=0.5,
                          random_state=42)

# --- Step 5: Train the Model ---
model.fit(X_train, y_train)

# --- Step 6: Predict on Test Data ---
y_pred = model.predict(X_test)

# --- Step 7: Evaluate the Model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", round(mse, 2))
print("R² Score:", round(r2, 3))

# --- Step 8: Display Each Weak Learner’s Performance ---
print("\n=== Performance of Each Weak Learner ===")
for i, learner in enumerate(model.estimators_):
    preds = learner.predict(X_test)
    mse_i = mean_squared_error(y_test, preds)
    print(f"Weak Learner {i+1}: MSE = {round(mse_i, 2)}")

# --- Step 9: Test on a New Sample ---
new_sample = np.array([[0.5, -1.2, 0.3, 2.1, -0.9, 0.8, 1.0, -0.4, 0.2, 1.3]])
predicted_value = model.predict(new_sample)

print("\n=== New Sample Test ===")
print("Input:", new_sample)
print("Predicted Value:", round(predicted_value[0], 2))

# --- Step 10: Display Model Weights and Errors ---
print("\nLearner Weights (αₜ):")
print(model.estimator_weights_)

print("\nLearner Errors (errₜ):")
print(model.estimator_errors_)
