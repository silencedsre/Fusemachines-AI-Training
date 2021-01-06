import mlflow
import pandas as pd
from config.config import DATA_PATH

from src.data.preprocess_data import preprocess_data
from src.features.vectorize import split_dataset, vectorize
from src.models.baseline import multinomial_nv_clf

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train_vec, X_test_vec = vectorize(X_train, X_test)
    with mlflow.start_run():
        clf, score, clf_rep = multinomial_nv_clf(
            X_train_vec, X_test_vec, y_train, y_test
        )
        mlflow.log_metric("score", score)
        mlflow.sklearn.log_model(clf, "sk_models")
