import numpy as np
import pandas as pd
from config.config import DATA_PATH
from src.data.preprocess_data import preprocess_data
from src.features.vectorize import split_dataset, vectorize
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import mlflow

def randomforest_clf(X_train_vec, X_test_vec, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    with mlflow.start_run() as run:
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        micro_p_r_f1_s=precision_recall_fscore_support(y_test, y_pred, average='micro')
        macro_p_r_f1_s = precision_recall_fscore_support(y_test, y_pred, average='macro')
        weighted_p_r_f1_s = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        clf_rep = classification_report(y_test, y_pred)
        #micro average value
        mlflow.log_metric("micro precision", micro_p_r_f1_s[0])
        mlflow.log_metric("micro recall", micro_p_r_f1_s[1])
        mlflow.log_metric("micro f1 score", micro_p_r_f1_s[2])
        # macro average value
        mlflow.log_metric("macro precision", macro_p_r_f1_s[0])
        mlflow.log_metric("macro recall", macro_p_r_f1_s[1])
        mlflow.log_metric("macro f1 score", macro_p_r_f1_s[2])
        # weighted average value
        mlflow.log_metric("weighted precision", weighted_p_r_f1_s[0])
        mlflow.log_metric("weighted recall", weighted_p_r_f1_s[1])
        mlflow.log_metric("weighted f1 score", weighted_p_r_f1_s[2])
    score = f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))
    return score, clf_rep

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train_vec, X_test_vec = vectorize(X_train, X_test)
    score, clf_rep = randomforest_clf(X_train_vec, X_test_vec, y_train, y_test)
    print(clf_rep)