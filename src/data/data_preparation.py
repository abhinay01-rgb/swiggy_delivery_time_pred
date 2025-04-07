import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

TARGET = "time_taken"

# Create logger
logger = logging.getLogger("data_preparation")
logger.setLevel(logging.INFO)

# Console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Formatter and handler setup
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded from {data_path}")
        logger.info(f"Data shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"The file at {data_path} does not exist.")
        raise

def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    logger.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    return train_data, test_data

def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path, index=False)
    logger.info(f"Data saved to {save_path}")

if __name__ == "__main__":
    # Root path
    root_path = Path(__file__).parent.parent.parent
    
    # File paths
    data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"
    save_data_dir = root_path / "data" / "interim"
    save_data_dir.mkdir(exist_ok=True, parents=True)

    # File names
    train_filename = "train.csv"
    test_filename = "test.csv"
    save_train_path = save_data_dir / train_filename
    save_test_path = save_data_dir / test_filename

    # Set parameters manually
    test_size = 0.2
    random_state = 42

    # Load, split and save
    df = load_data(data_path)
    train_data, test_data = split_data(df, test_size, random_state)

    for filename, path, data in zip([train_filename, test_filename], [save_train_path, save_test_path], [train_data, test_data]):
        save_data(data, path)
