import pathlib
import pickle
import mlflow
import pandas as pd
from config.config import BASE_DIR, DATA_PATH, VECTORIZER_PATH, MODEL_PATH

from src.data.preprocess_data import preprocess_data
from src.features.vectorize import split_dataset, fit_vectorizer, transform_vectorizer
from src.models.baseline import randomforest_clf

# model_path = "models/randomforest/model.pkl"  # TODO change path from config


def save_trained_model(vect=None):
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    vect = fit_vectorizer(X_train)
    X_train_vec = transform_vectorizer(vect, X_train)
    X_test_vec = transform_vectorizer(vect, X_test)

    with mlflow.start_run():
        clf, score, clf_rep = randomforest_clf(
            X_train_vec, X_test_vec, y_train, y_test
        )
        mlflow.sklearn.save_model(
            clf,
            path=MODEL_PATH.parent,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )  # TODO change path from config


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        save_trained_model()

    vect = fit_vectorizer()

    input = ["think short time live"]
    inp_vec = transform_vectorizer(vectorizer=vect, data=input)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    pred = model.predict_proba(inp_vec)
    print(pred)
