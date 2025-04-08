import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn import set_config
import joblib

# Output as DataFrame
set_config(transform_output='pandas')

# === Column Types ===
num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]
nominal_cat_cols = ['weather', 'type_of_order', 'type_of_vehicle', "festival", "city_type", "is_weekend", "order_time_of_day"]
ordinal_cat_cols = ["traffic", "distance_type"]
target_col = "time_taken"

# === Ordinal Encoding Orders ===
traffic_order = ["low", "medium", "high", "jam"]
distance_type_order = ["short", "medium", "long", "very_long"]

# === Paths ===
root_path = Path(__file__).parent.parent.parent
train_path = root_path / "data" / "interim" / "train.csv"
test_path = root_path / "data" / "interim" / "test.csv"
save_dir = root_path / "data" / "processed"
save_dir.mkdir(exist_ok=True, parents=True)

# === Load and Clean Data ===
train_df = pd.read_csv(train_path).dropna()
test_df = pd.read_csv(test_path).dropna()

# === Split Features & Target ===
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# === Preprocessor ===
preprocessor = ColumnTransformer([
    ("scale", MinMaxScaler(), num_cols),
    ("nominal", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), nominal_cat_cols),
    ("ordinal", OrdinalEncoder(categories=[traffic_order, distance_type_order],
                               encoded_missing_value=-999,
                               handle_unknown="use_encoded_value",
                               unknown_value=-1), ordinal_cat_cols)
], remainder="passthrough", verbose_feature_names_out=False)

# === Fit and Transform ===
preprocessor.fit(X_train)
X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# === Save Transformed Data ===
train_trans_df = X_train_trans.join(y_train)
test_trans_df = X_test_trans.join(y_test)
train_trans_df.to_csv(save_dir / "train_trans.csv", index=False)
test_trans_df.to_csv(save_dir / "test_trans.csv", index=False)


# === Save the Preprocessor ===
model_dir = root_path / "models"
model_dir.mkdir(exist_ok=True)  # Ensure models directory exists
joblib.dump(preprocessor, model_dir / "preprocessor.pkl")

print("Preprocessor saved to:", model_dir / "preprocessor.pkl")



