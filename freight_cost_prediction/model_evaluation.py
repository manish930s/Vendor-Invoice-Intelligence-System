import math
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_regression(x_train, y_train):
  model = LinearRegression()
  model.fit(x_train, y_train)
  return model

def train_decision_tree(x_train, y_train):
  model = DecisionTreeRegressor(max_depth=5, random_state=42)
  model.fit(x_train, y_train)
  return model

def train_random_forest(x_train, y_train):
  model = RandomForestRegressor(max_depth=5, random_state=42)
  model.fit(x_train, y_train)
  return model

def evaluate_model(model, x_test, y_test, model_name: str) -> dict:
  """
  Evaluate regression model and return performance metrics
  """
  preds = model.predict(x_test)
  mae = mean_absolute_error(y_test, preds)
  mse = math.sqrt(mean_squared_error(y_test, preds))
  r2 = r2_score(y_test, preds) * 100

  print(f"\n{model_name} Performance:")
  print(f"MAE: {mae:.2f}")
  print(f"RMSE: {mse:.2f}")
  print(f"R2: {r2:.2f}")

  return {
      'model_name': model_name,
      'mae': mae,
      'mse': mse,
      'r2': r2
  }