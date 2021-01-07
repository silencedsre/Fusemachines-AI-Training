import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.data.preprocess_data import preprocess_data
from config.config import BASE_DIR, DATA_PATH, VECTORIZER_PATH


def split_dataset(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["emotion"], test_size=0.2, random_state=42
    )  # TODO: Iterative-Stratification
    return X_train.tolist(), X_test.tolist(), y_train, y_test


def fit_vectorizer(X_train=None):
    if not VECTORIZER_PATH.exists():
        temp = Path(BASE_DIR).joinpath("temp")
        temp.mkdir(exist_ok=True)
        vectorizer = TfidfVectorizer().fit(X_train)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)

    with open(VECTORIZER_PATH, "rb") as f:
        vect = pickle.load(f)
    return vect


def transform_vectorizer(vectorizer, data):
    vect = vectorizer.transform(data)
    return vect.toarray()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    vect = fit_vectorizer(X_train)
    # with open(VECTORIZER_PATH, "rb") as f:
    #     vect = pickle.load(f)

    X_train_vec = transform_vectorizer(vectorizer=vect, data=X_train)
    X_test_vec = transform_vectorizer(vectorizer=vect, data=X_test)
    print(X_train_vec.shape)
