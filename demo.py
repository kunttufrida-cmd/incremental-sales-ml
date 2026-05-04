# XGBoost demo for incremental sales prediction
# data generation → training → prediction → evaluation
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Generate fake data
np.random.seed(42)

n = 100
time = np.arange(n)
promotion = np.random.randint(0, 2, n)
sales = 100 + 0.5 * time + 20 * promotion + np.random.normal(0, 5, n)

# 2. Create features and target
X = np.column_stack((time, promotion))
y = sales

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create model
model = XGBRegressor(objective="reg:squarederror")

# 5. Train model
model.fit(X_train, y_train)

# 6. Predict
predictions = model.predict(X_test)

# 7. Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("First 5 predictions:", predictions[:5])
print("Actual values: ",y_test[:5])