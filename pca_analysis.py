import os
import pickle
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


def load_data():
    x_train = pd.read_csv("data/processed/X_train.csv")
    x_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    return x_train, x_test, y_train


def run_pca(x_train, x_test, n_components=50):
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca, pca


def main():
    os.makedirs("data/processed", exist_ok=True)

    x_train, x_test, _ = load_data()

    x_train_pca, x_test_pca, pca = run_pca(x_train, x_test)

    cols = [f"pc{i+1}" for i in range(x_train_pca.shape[1])]

    pd.DataFrame(x_train_pca, columns=cols).to_csv(
        "data/processed/X_train_pca.csv", index=False
    )
    pd.DataFrame(x_test_pca, columns=cols).to_csv(
        "data/processed/X_test_pca.csv", index=False
    )

    with open("data/processed/pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)


if __name__ == "__main__":
    main()
