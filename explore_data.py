import pandas as pd


def load_data():
    expression = pd.read_csv("data/raw/expression_matrix.csv")
    labels = pd.read_csv("data/raw/sample_labels.csv")
    return expression, labels


def find_label_column(labels):
    for col in labels.columns:
        if col.lower() in ["class", "label", "type", "cancer_type", "cancer"]:
            return col
    return labels.columns[0]


def main():
    expression, labels = load_data()
    label_col = find_label_column(labels)

    print("expression shape:", expression.shape)
    print("labels shape:", labels.shape)
    print("label column:", label_col)

    counts = labels[label_col].value_counts()
    print("\nclass distribution:")
    print(counts)

    print("\nmissing values:")
    print("expression:", expression.isnull().sum().sum())
    print("labels:", labels.isnull().sum().sum())

    if len(expression) != len(labels):
        print("warning: sample count mismatch")


if __name__ == "__main__":
    main()
