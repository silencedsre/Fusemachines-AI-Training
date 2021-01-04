import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.data.preprocess_data import preprocess_data
from config.config import DATA_PATH

def split_dataset(df):
    X_train, X_test, y_train, y_test = train_test_split(\
        df['review'], df['emotion'], test_size=0.2)  #TODO: Iterative-Stratification
    return X_train, X_test, y_train, y_test

def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec.toarray(), X_test_vec.toarray()

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train_vec, X_test_vec = vectorize(X_train, X_test)
    print(X_train_vec)
