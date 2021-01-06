import pickle
import numpy as np
import pandas as pd
from config.config import DATA_PATH
from src.data.preprocess_data import preprocess_data
from src.features.vectorize import split_dataset, fit_vectorizer, transform_vectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


def multinomial_nv_clf(X_train_vec, X_test_vec, y_train, y_test):
    clf = OneVsRestClassifier(MultinomialNB())
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    score = f1_score(y_test, y_pred, average="macro", labels=np.unique(y_pred))
    clf_rep = classification_report(y_test, y_pred)
    return clf, score, clf_rep


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    with open("../../temp/vectorizer", "rb") as f:  # TODO change path from config
        vect = pickle.load(f)

    X_train_vec = transform_vectorizer(vectorizer=vect, data=X_train)
    X_test_vec = transform_vectorizer(vectorizer=vect, data=X_test)
    clf, score, clf_rep = multinomial_nv_clf(X_train_vec, X_test_vec, y_train, y_test)

    input = ["I love apples"]
    vec = transform_vectorizer(vectorizer=vect, data=input)
    pred = clf.predict_proba(vec)
    print(list(enumerate(pred.ravel().tolist())))
