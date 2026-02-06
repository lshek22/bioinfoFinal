import os
import time
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_data():
    x_train = pd.read_csv("data/processed/X_train_pca.csv")
    x_test = pd.read_csv("data/processed/X_test_pca.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    return x_train, x_test, y_train, y_test, encoder


def get_models():
    return {
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(30, 15),
            max_iter=500,
            early_stopping=True,
            random_state=42
        ),
        "svm": SVC(kernel="rbf", C=10, probability=True, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, n_jobs=-1, random_state=42
        )
    }


def evaluate(model, x_train, y_train, x_test, y_test, encoder):
    start = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv)

    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]

    y_test_bin = (encoder.inverse_transform(y_test) == "Non-LUAD").astype(int)

    return {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test_bin, probs),
        "training_time": train_time
    }


def main():
    os.makedirs("results/metrics", exist_ok=True)

    x_train, x_test, y_train, y_test, encoder = load_data()
    models = get_models()

    results = []

    for name, model in models.items():
        metrics = evaluate(
            model, x_train, y_train, x_test, y_test, encoder
        )
        metrics["model"] = name
        results.append(metrics)

    df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    df.to_csv("results/metrics/model_comparison.csv", index=False)

    print(df)


if __name__ == "__main__":
    main()
