import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Target column name
TARGET = "time_taken"

# Load the test data
def load_data(path):
    return pd.read_csv(path)

# Split into features and target
def make_X_and_y(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

if __name__ == "__main__":
    # Paths
    root_path = Path(__file__).parent.parent.parent
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    model_path = root_path / "models" / "model.pkl"

    # Load test data and model
    test_data = load_data(test_data_path)
    model = joblib.load(model_path)

    # Split into features and target
    X_test, y_test = make_X_and_y(test_data, TARGET)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate and print evaluation metrics
    print("Model Evaluation Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
