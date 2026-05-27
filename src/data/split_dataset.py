import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df, label_column="label"):
    train_df = []
    test_df = []

    for label in df[label_column].unique():
        subset = df[df[label_column] == label]

        train, test = train_test_split(
            subset,
            train_size=400,
            test_size=100,
            random_state=42,
            shuffle=True
        )

        train_df.append(train)
        test_df.append(test)

    train_df = pd.concat(train_df)
    test_df = pd.concat(test_df)

    return train_df, test_df