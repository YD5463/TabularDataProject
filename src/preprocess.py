import numpy as np

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.io.arff import loadarff
from sklearn.datasets import load_digits, load_iris


def load_mnist(nan_percentage: float):
    MISSING_FEATURE = 28
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
    # print(df["Embarked"].value_counts())
    # mask = df["Embarked"] == 'Q'
    # nan_indices = np.random.uniform(size=df[mask]["Age"].shape[0]) < nan_percentage
    nan_indices = np.random.uniform(size=df["Age"].shape[0]) < nan_percentage
    y_true = df["Age"].values.copy()
    df["Age"][nan_indices] = np.nan
    # df[mask][nan_indices]["Age"] = np.nan
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, y_true


def load_house_pricing():
    df = pd.read_csv("more_data/house_pricing/train.csv")
    y = df["SalePrice"]
    X = df.drop(["SalePrice", "Id"], axis=1)
    # nan_count = X.isna().sum() / X.shape[0]
    # np.percentile(nan_count,95)
    # X = X[X.columns[nan_count < 0.9]]
    very_num_cols = X._get_numeric_data().columns
    categorical_cols = list(set(X.columns) - set(very_num_cols))
    # X = X.drop(s[s].index, axis=1)
    X = pd.get_dummies(X, columns=categorical_cols)
    return X


def load_data():
    raw_data = loadarff('more_data/2d-10c.arff')
    df = pd.DataFrame(raw_data[0])
    df["CLASS"] = df["CLASS"].astype(np.float16)
    missing_values = df["CLASS"].values
    df = df.drop("CLASS", axis=1)
    return df, missing_values


# #https://www.kaggle.com/datasets/thedevastator/clustering-polygons-utilizing-iris-moon-and-circ
# def load_galaxy(nan_percentage: float):
#     df = pd.concat([
#         pd.read_csv("more_data/galaxy/circles.csv"),
#         pd.read_csv("more_data/galaxy/iris.csv"),
#         pd.read_csv("more_data/galaxy/moon.csv")
#     ])
#     return df


def load_iris_dataset(nan_percentage: float):
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],columns=iris['feature_names'] + ['target'])
    NAN_COLUMN = "petal width (cm)"
    nan_indices = np.random.uniform(size=df[NAN_COLUMN].shape[0]) < nan_percentage
    y_true = df[NAN_COLUMN].values.copy()
    df[NAN_COLUMN][nan_indices] = np.nan
    return df, y_true
