import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.data.preprocess_data import preprocess_data
from config.config import DATA_PATH


def split_dataset(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["emotion"], test_size=0.2
    )  # TODO: Iterative-Stratification
    return X_train.tolist(), X_test.tolist(), y_train, y_test


def fit_vectorizer(X_train):
    vectorizer = TfidfVectorizer().fit(X_train)
    return vectorizer


def transform_vectorizer(vectorizer, data):
    vect = vectorizer.transform(data)
    return vect.toarray()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    vect = fit_vectorizer(X_train)
    X_train_vec = transform_vectorizer(vectorizer=vect, data=X_train)
    X_test_vec = transform_vectorizer(vectorizer=vect, data=X_test)
    print(X_train_vec.shape)
