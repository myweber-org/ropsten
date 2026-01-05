import pandas as pd
import numpy as np
import re

def clean_column_names(df):
    """
    Standardize column names: lowercase, replace spaces with underscores, remove special characters.
    """
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)
        col_str = col_str.lower().strip()
        col_str = re.sub(r'\s+', '_', col_str)
        col_str = re.sub(r'[^a-z0-9_]', '', col_str)
        cleaned_columns.append(col_str)
    df.columns = cleaned_columns
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def fill_missing_values(df, numeric_strategy='mean', categorical_strategy='mode'):
    """
    Fill missing values in numeric columns with mean/median and categorical columns with mode.
    """
    df_filled = df.copy()
    
    for column in df_filled.columns:
        if df_filled[column].dtype in ['int64', 'float64']:
            if numeric_strategy == 'mean':
                fill_value = df_filled[column].mean()
            elif numeric_strategy == 'median':
                fill_value = df_filled[column].median()
            else:
                fill_value = 0
            df_filled[column].fillna(fill_value, inplace=True)
        else:
            if categorical_strategy == 'mode' and not df_filled[column].mode().empty:
                fill_value = df_filled[column].mode()[0]
            else:
                fill_value = 'Unknown'
            df_filled[column].fillna(fill_value, inplace=True)
    
    return df_filled

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a numeric column using the Interquartile Range (IQR) method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def standardize_numeric(df, columns):
    """
    Standardize numeric columns to have zero mean and unit variance.
    """
    df_standardized = df.copy()
    for col in columns:
        if col in df_standardized.columns and df_standardized[col].dtype in ['int64', 'float64']:
            mean = df_standardized[col].mean()
            std = df_standardized[col].std()
            if std > 0:
                df_standardized[col] = (df_standardized[col] - mean) / std
    return df_standardized

def clean_csv_file(input_path, output_path, cleaning_steps=None):
    """
    Main function to apply a series of cleaning steps to a CSV file.
    """
    df = pd.read_csv(input_path)
    
    if cleaning_steps is None:
        cleaning_steps = [
            ('clean_column_names', {}),
            ('remove_duplicates', {'subset': None, 'keep': 'first'}),
            ('fill_missing_values', {'numeric_strategy': 'mean', 'categorical_strategy': 'mode'})
        ]
    
    for step, kwargs in cleaning_steps:
        if step == 'clean_column_names':
            df = clean_column_names(df)
        elif step == 'remove_duplicates':
            df = remove_duplicates(df, **kwargs)
        elif step == 'fill_missing_values':
            df = fill_missing_values(df, **kwargs)
        elif step == 'remove_outliers_iqr':
            df = remove_outliers_iqr(df, **kwargs)
        elif step == 'standardize_numeric':
            df = standardize_numeric(df, **kwargs)
    
    df.to_csv(output_path, index=False)
    return df