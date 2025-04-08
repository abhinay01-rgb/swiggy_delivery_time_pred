import pandas as pd
import joblib
import yaml
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor

# Target column name
TARGET = "time_taken"

# Load data
def load_data(path):
    return pd.read_csv(path)

# Split into features and target
def make_X_and_y(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# Train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Save the model
def save_model(model, save_path):
    joblib.dump(model, save_path)

# Read hyperparameters from YAML
def read_params(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

# Get the right model based on config
def get_model(model_name, params):
    if model_name == "XGBoost":
        return XGBRegressor(**params)
    elif model_name == "RandomForest":
        return RandomForestRegressor(**params)
    elif model_name == "LightGBM":
        return LGBMRegressor(**params)
    elif model_name == "DecisionTree":
        return DecisionTreeRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

if __name__ == "__main__":
    # Paths
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "processed" / "train_trans.csv"
    model_save_path = root_path / "models" / "model.pkl"
    params_path = root_path / "params.yaml"

    # Create models folder if not exists
    model_save_path.parent.mkdir(exist_ok=True)

    # Load data
    data = load_data(data_path)
    X_train, y_train = make_X_and_y(data, TARGET)

    # Load parameters
    params = read_params(params_path)
    model_name = params["Train"]["model_name"]
    model_params = params["Train"][model_name]

    # Create and train the model
    model = get_model(model_name, model_params)
    trained_model = train_model(model, X_train, y_train)

    # Save the model
    save_model(trained_model, model_save_path)

    print(f"{model_name} model trained and saved to:", model_save_path)
