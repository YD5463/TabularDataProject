import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_titanic():
    df_train = pd.read_csv("./titanic/train.csv")
    df_test = pd.read_csv("./titanic/test.csv")
    df = pd.concat([df_test, df_train])
    df = df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
    df = df[df['Survived'].notna()]
    df = df[df['Embarked'].notna()]
    df = df[df["Age"].notna()]
    missing_values = df["Age"].values
    df = df.drop("Age", axis=1)
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, missing_values


def load_house_pricing():
    df = pd.read_csv("house_pricing/train.csv")
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
