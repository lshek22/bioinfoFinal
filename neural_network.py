import os
import pickle
import time
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data():
    x_train = pd.read_csv("data/processed/X_train_pca.csv")
    x_test = pd.read_csv("data/processed/X_test_pca.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    return x_train, x_test, y_train, y_test, encoder


def build_model():
    return MLPClassifier(
        hidden_layer_sizes=(30, 15),
        max_iter=500,
        early_stopping=True,
        random_state=42
    )


def main():
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    x_train, x_test, y_train, y_test, encoder = load_data()
    model = build_model()

    start = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv)

    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]

    y_test_bin = (encoder.inverse_transform(y_test) == "Non-LUAD").astype(int)

    metrics = {
        "test_accuracy": accuracy_score(y_test, preds),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "roc_auc": roc_auc_score(y_test_bin, probs),
        "training_time": train_time
    }

    pd.DataFrame([metrics]).to_csv(
        "results/metrics/neural_network_metrics.csv", index=False
    )

    with open("results/models/neural_network.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
