import json
import pickle
import mlflow
import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from config.config import BASE_DIR, DATA_PATH, VECTORIZER_PATH, MODEL_PATH

from src.data.preprocess_data import preprocess_data
from src.features.vectorize import split_dataset, fit_vectorizer, transform_vectorizer
from src.models.baseline import multinomial_nv_clf


app = Flask(__name__)

CORS(app)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "Prediction"

    if request.method == "POST":
        text = request.get_data()
        inp = [text]
        if not MODEL_PATH.exists():
            save_trained_model()

        vect = fit_vectorizer()

        input = ["think short time live"]
        inp_vec = transform_vectorizer(vectorizer=vect, data=inp)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        pred = model.predict_proba(inp_vec)
        pred = pred.ravel().tolist()
        print(pred)
        return json.dumps({"pred": pred})


def save_trained_model(vect=None):
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    vect = fit_vectorizer(X_train)
    X_train_vec = transform_vectorizer(vect, X_train)
    X_test_vec = transform_vectorizer(vect, X_test)

    with mlflow.start_run():
        clf, score, clf_rep = multinomial_nv_clf(
            X_train_vec, X_test_vec, y_train, y_test
        )
        mlflow.sklearn.save_model(
            clf,
            path=MODEL_PATH.parent,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )


if __name__ == "__main__":
    app.run(debug=True)

    # if not MODEL_PATH.exists():
    #     save_trained_model()
    #
    # vect = fit_vectorizer()
    #
    # input = ["think short time live"]
    # inp_vec = transform_vectorizer(vectorizer=vect, data=input)
    # with open(MODEL_PATH, "rb") as f:
    #     model = pickle.load(f)
    # pred = model.predict_proba(inp_vec)
    # print(pred)
