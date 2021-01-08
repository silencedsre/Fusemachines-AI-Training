from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(BASE_DIR).joinpath("data")
Path(DATA_DIR).mkdir(exist_ok=True)
DATA_PATH = Path.joinpath(BASE_DIR, "data", "ISEAR.csv")
VECTORIZER_PATH = Path.joinpath(BASE_DIR, "temp", "vectorizer")
MODEL_PATH = Path.joinpath(BASE_DIR, "models", "naive_bayes", "model.pkl")

DATABASE_USERNAME = ""
DATABASE_PASSWORD = ""
DATABASE_HOSTNAME = ""

# Hyperparameters
