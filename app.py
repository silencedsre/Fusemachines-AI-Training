import json
import pickle
import mlflow
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mongoengine import MongoEngine
from config.config import BASE_DIR, DATA_PATH, VECTORIZER_PATH, MODEL_PATH

from src.data.preprocess_data import preprocess_data
from src.features.vectorize import split_dataset, fit_vectorizer, transform_vectorizer
from src.models.baseline import multinomial_nv_clf


app = Flask(__name__)
app.config["MONGODB_SETTINGS"] = {  # TODO get from config
    "db": "fusemachines_ai_training_2",
    "host": "db",
    "port": 27017,
}
db = MongoEngine()
db.init_app(app)

app = Flask(__name__)

CORS(app)


class Review(db.Document):
    comment = db.StringField()
    prediction = db.DictField()

    def to_json(self):
        return {"comment": self.comment, "prediction": self.prediction}


@app.route("/", methods=["GET"])
def query_records():
    comment_data = []
    pred = []
    review = Review.objects()
    for rev in review:
        comment_data.append(rev.comment)
        pred.append(rev.prediction)
    data = list(zip(comment_data, pred))
    return jsonify({"data": data})


@app.route("/", methods=["POST"])
def update_record():
    record = json.loads(request.data)
    review_text = record["comment"]
    input = [review_text]
    if not MODEL_PATH.exists():
        save_trained_model()

    vect = fit_vectorizer()

    inp_vec = transform_vectorizer(vectorizer=vect, data=input)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    pred = model.predict_proba(inp_vec)
    pred = pred.ravel().tolist()
    classes = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]
    pred = {classes[i]: pred[i] for i in range(len(classes))}
    review = Review(comment=record["comment"], prediction=pred)
    review.save()
    return json.dumps({"pred": pred})


def save_trained_model():
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
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(host='0.0.0.0', port:5000, debug=True)
