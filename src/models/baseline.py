import numpy as np
import pandas as pd
from config.config import DATA_PATH
from src.data.preprocess_data import preprocess_data
from src.features.vectorize import split_dataset, vectorize
from sklearn.metrics import f1_score, classification_report
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

def randomforest_clf(X_train_vec, X_test_vec, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    # clf = OneVsRestClassifier(MultinomialNB())
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    score = f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))
    clf_rep = classification_report(y_test, y_pred)
    return score, clf_rep

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train_vec, X_test_vec = vectorize(X_train, X_test)
    score, clf_rep = randomforest_clf(X_train_vec, X_test_vec, y_train, y_test)
    print(clf_rep)