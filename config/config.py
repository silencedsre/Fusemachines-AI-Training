from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path.joinpath(BASE_DIR, "data", "ISEAR.csv")
VECTORIZER_PATH = Path.joinpath(BASE_DIR, "temp", "vectorizer")
MODEL_PATH = Path.joinpath(BASE_DIR, "models", "randomforest", "model.pkl")

DATABASE_USERNAME = ""
DATABASE_PASSWORD = ""
DATABASE_HOSTNAME = ""

# Hyperparameters