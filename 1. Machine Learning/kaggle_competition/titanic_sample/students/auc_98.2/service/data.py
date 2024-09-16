# 데이터 불러오는 모듈
import pandas as pd


def load_data():
    # train = pd.read_csv('./data/train.csv')
    train = pd.read_csv("./data/train.csv")

    # 위험할 때 사용할 것
    # train_kaggle_df = pd.read_csv("./data/train_kaggle.csv")
    # train_kaggle_df.columns = train_kaggle_df.columns.str.lower().str.replace(
    #     "sex", "gender"
    # )

    # train = pd.concat([train_df, train_kaggle_df]).drop_duplicates()
    test = pd.read_csv("./data/test.csv")
    submission = pd.read_csv("./data/submission.csv")

    return train, test, submission
