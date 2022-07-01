import pandas as pd 
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")

    df["kfold"] = -1

    df = df.sample(frac=0.1).reset_index(drop=True)

    label = df.label.values

    kf = StratifiedKFold(n_splits=2)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=label)):
        df.loc[v_, "kfold"] = f

    df.to_csv("data/train_fold.csv", index=False)