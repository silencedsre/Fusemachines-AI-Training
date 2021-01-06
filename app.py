import mlflow
import pickle
import pandas as pd
from config.config import DATA_PATH

from src.data.preprocess_data import preprocess_data
from src.features.vectorize import split_dataset, fit_vectorizer, transform_vectorizer
from src.models.baseline import multinomial_nv_clf


model_path = "/home/shree/FuseWork/AI engineers Training/Fusemachines-AI-Training/models/naive_bayes/model.pkl"  # TODO change path from config

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    with open("temp/vectorizer", "rb") as f:  # TODO change path from config
        vect = pickle.load(f)
    X_train_vec = transform_vectorizer(vect, X_train)
    X_test_vec = transform_vectorizer(vect, X_test)

    with mlflow.start_run():
        clf, score, clf_rep = multinomial_nv_clf(
            X_train_vec, X_test_vec, y_train, y_test
        )
        mlflow.sklearn.save_model(
            clf,
            path="models/naive_bayes",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )  # TODO change path from config

    input = ["think short time live"]
    inp_vec = transform_vectorizer(vectorizer=vect, data=input)
    print(inp_vec.shape)
    # clf = mlflow.sklearn.load_model(model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    pred = model.predict_proba(inp_vec)
    print(pred)
