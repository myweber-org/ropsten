import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and dropping specified columns.
    
    Args:
        filepath (str): Path to the CSV file
        missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns_to_drop (list): List of column names to drop from the dataset
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(filepath)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if df[column].isnull().any():
                if missing_strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif missing_strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif missing_strategy == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif missing_strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
        
        for column in df.select_dtypes(exclude=[np.number]).columns:
            if df[column].isnull().any():
                df[column].fillna('Unknown', inplace=True)
        
        df = df.reset_index(drop=True)
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers in a numerical column using IQR method.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pandas.DataFrame: Rows identified as outliers
    """
    if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
        return pd.DataFrame()
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned dataframe
        output_path (str): Path to save the cleaned CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False