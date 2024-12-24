import pandas as pd
from sklearn import model_selection
import os


def get_fold(root):
    csv_path = os.path.join(root, 'train.csv')
    df = pd.read_csv(csv_path)

    df["kfold"] = -1

    df = df.sample(frac=0.1).reset_index(drop=True)

    label = df.label.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=label)):
        df.loc[v_, "kfold"] = f

    df.to_csv("data/train_fold.csv", index=False)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")

    df["kfold"] = -1

    df = df.sample(frac=0.1).reset_index(drop=True)

    label = df.label.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=label)):
        df.loc[v_, "kfold"] = f

    df.to_csv("data/train_fold.csv", index=False)
