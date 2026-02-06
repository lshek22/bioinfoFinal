import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    x = pd.read_csv("data/raw/expression_matrix.csv")
    y = pd.read_csv("data/raw/sample_labels.csv")
    return x, y


def build_binary_labels(labels):
    labels["binary_class"] = labels["Class"].apply(
        lambda v: "LUAD" if v == "LUAD" else "Non-LUAD"
    )
    return labels["binary_class"]


def split_and_scale(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    x_train = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x.columns,
        index=x_train.index
    )
    x_test = pd.DataFrame(
        scaler.transform(x_test),
        columns=x.columns,
        index=x_test.index
    )

    return x_train, x_test, y_train, y_test, scaler


def main():
    os.makedirs("data/processed", exist_ok=True)

    x, labels = load_data()

    if "Unnamed: 0" in x.columns:
        x = x.drop(columns=["Unnamed: 0"])

    y = build_binary_labels(labels)

    x_train, x_test, y_train, y_test, scaler = split_and_scale(x, y)

    x_train.to_csv("data/processed/X_train.csv", index=False)
    x_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
