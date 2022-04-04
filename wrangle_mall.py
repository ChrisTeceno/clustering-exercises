import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from env import get_db_url
import os
from matplotlib import style

style.use("ggplot")
from sklearn.model_selection import train_test_split


def get_mall_data(use_cache=True):
    """pull from SQL unless mall_data.csv exists"""
    filename = "mall_data.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("mall_customers")
    query = """
    SELECT *
    From customers"""
    mall_df = pd.read_sql(query, url, index_col="customer_id")
    print("Saving to csv in local directory...")
    mall_df.to_csv(filename, index=False)
    return mall_df


def basic_info(df):
    """print some basic information about the dataframe"""
    print(df.info())
    print(df.describe())
    print("\n")
    print("null counts:")
    print(df.isnull().sum())


def detect_outliers(df, col, k=1.5):
    """look for outliers in a column of a dataframe using IQR, k"""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr * k)
    upper_bound = q3 + (iqr * k)
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]


def remove_outliers(df, k=1.5):
    """remove outliers from all quantitative variables"""
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df = df[~df[col].index.isin(detect_outliers(df, col, k).index)]
    return df


def encode_categoricals(df):
    """encode categorical variables using dummy variables"""
    for col in df.select_dtypes(include=["object"]).columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    return df


def split_data(df):
    """split data into train, validate and test sets"""
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, validate = train_test_split(train, test_size=0.3, random_state=42)
    return train, validate, test


def scale_numerical(train, validate, test, scaler=StandardScaler()):
    """scale numerical variables after fitting on training set"""
    # choose numerical columns
    numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns
    # fit and transform training data
    train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
    # transform validation data
    validate[numerical_cols] = scaler.transform(validate[numerical_cols])
    # transform test data
    test[numerical_cols] = scaler.transform(test[numerical_cols])
    return train, validate, test


def wrangle(scaler=StandardScaler(), k=1.5):
    """wrangle data"""
    df = get_mall_data()
    df = remove_outliers(df)
    df = encode_categoricals(df)
    train, validate, test = split_data(df)
    train, validate, test = scale_numerical(train, validate, test)
    return train, validate, test
