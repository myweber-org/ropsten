import pandas as pd
import numpy as np

def remove_missing_values(df, threshold=0.5):
    """
    Remove columns with missing values above threshold.
    """
    missing_percent = df.isnull().sum() / len(df)
    columns_to_drop = missing_percent[missing_percent > threshold].index
    return df.drop(columns=columns_to_drop)

def normalize_numeric_columns(df, columns=None):
    """
    Normalize specified numeric columns to range [0,1].
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = columns
    
    for col in numeric_cols:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df

def encode_categorical(df, columns=None, method='onehot'):
    """
    Encode categorical columns using specified method.
    """
    if columns is None:
        categorical_cols = df.select_dtypes(include=['object']).columns
    else:
        categorical_cols = columns
    
    if method == 'onehot':
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
    
    return df

def clean_dataset(df, missing_threshold=0.5, normalize=True, encode=True):
    """
    Main cleaning pipeline combining all steps.
    """
    df_clean = df.copy()
    
    df_clean = remove_missing_values(df_clean, missing_threshold)
    
    if normalize:
        df_clean = normalize_numeric_columns(df_clean)
    
    if encode:
        df_clean = encode_categorical(df_clean)
    
    return df_clean