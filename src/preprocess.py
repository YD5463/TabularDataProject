import numpy as np

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits


def load_mnist(nan_percentage: float):
    MISSING_FEATURE = 35
    df = pd.DataFrame(load_digits().data)
    nan_indices = np.random.uniform(size=df[MISSING_FEATURE].shape[0]) < nan_percentage
    y_true = df[MISSING_FEATURE].values.copy()
    df[MISSING_FEATURE][nan_indices] = np.nan
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, y_true


def load_titanic(nan_percentage):
    df_train = pd.read_csv("./titanic/train.csv")
    df_test = pd.read_csv("./titanic/test.csv")
    df = pd.concat([df_test, df_train])
    df = df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
    df = df[df['Survived'].notna()]
    df = df[df['Embarked'].notna()]
    df = df[df["Age"].notna()]
    print(df["Embarked"].value_counts())
    # nan_indices = np.random.uniform(size=df["Age"].shape[0]) < nan_percentage
    y_true = df["Age"].values.copy()
    # df["Age"][nan_indices] = np.nan
    nan_mask = np.random.uniform(size=df["Embarked"].shape[0]) < 0.8
    df["Age"][(df["Embarked"] == 'C') & nan_mask] = np.nan
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, y_true


def load_mobile_price_dataset(nan_percentage: float):
    df = pd.concat([pd.read_csv("./datasets/mobile_price/test.csv"),
                    pd.read_csv("./datasets/mobile_price/train.csv")])
    df = df.drop(["id"],axis=1)
    df = df[df["price_range"].notna()]
    NAN_COLUMN = "battery_power"
    nan_indices = np.random.uniform(size=df[NAN_COLUMN].shape[0]) < nan_percentage
    y_true = df[NAN_COLUMN].values.copy()
    df[NAN_COLUMN][nan_indices] = np.nan
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, y_true
