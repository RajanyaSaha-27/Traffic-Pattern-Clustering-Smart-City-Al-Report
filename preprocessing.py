import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data(file_path):
    """
    Load CSV dataset
    """

    df = pd.read_csv(file_path)

    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"Shape : {df.shape}")

    return df


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

def dataset_summary(df):

    print("\nDataset Information")
    print("-" * 60)

    print(df.info())

    print("\nFirst 5 Rows")
    print(df.head())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows :", df.duplicated().sum())


# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------

def clean_data(df):

    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill numerical missing values
    numerical_columns = df.select_dtypes(include=np.number).columns

    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    categorical_columns = df.select_dtypes(include="object").columns

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def feature_engineering(df):

    df = df.copy()

    # # Convert Date column
    # if "Date" in df.columns:
    #     df["Date"] = pd.to_datetime(df["Date"], format="%d")

    #     df["Year"] = df["Date"].dt.year
    #     df["Month"] = df["Date"].dt.month
    #     df["Day"] = df["Date"].dt.day
    #     df["Weekday"] = df["Date"].dt.day_name()

    # Convert Time column
    if "Time" in df.columns:

        df["Time"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p")

        df["Hour"] = df["Time"].dt.hour

    return df


# --------------------------------------------------
# Label Encoding
# --------------------------------------------------

def encode_features(df):

    df = df.copy()

    encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(include="object").columns

    for col in categorical_columns:

        df[col] = encoder.fit_transform(df[col])

    return df


# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

def scale_features(df):

    df = df.copy()

    scaler = StandardScaler()

    numeric_columns = df.select_dtypes(include=np.number).columns

    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df, scaler


# --------------------------------------------------
# Complete Pipeline
# --------------------------------------------------

def preprocess_data(file_path):

    df = load_data(file_path)

    dataset_summary(df)

    df = clean_data(df)

    df = feature_engineering(df)

    encoded_df = encode_features(df)

    scaled_df, scaler = scale_features(encoded_df)

    print("\nPreprocessing Completed Successfully")

    return df, encoded_df, scaled_df, scaler