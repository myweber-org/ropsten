import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def remove_outliers(df, column, threshold=3):
    """Remove outliers using the Z-score method."""
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return df
    
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    filtered_df = df[z_scores < threshold]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'.")
    return filtered_df

def normalize_column(df, column):
    """Normalize a column to range [0, 1]."""
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return df
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        print(f"Column '{column}' has constant values. Normalization skipped.")
        return df
    
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    print(f"Column '{column}' normalized successfully.")
    return df

def clean_data(df, numeric_columns, outlier_threshold=3):
    """Main data cleaning function."""
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return df
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers(cleaned_df, col, outlier_threshold)
            cleaned_df = normalize_column(cleaned_df, col)
        else:
            print(f"Skipping column '{col}' as it does not exist.")
    
    print(f"Data cleaning completed. Final shape: {cleaned_df.shape}")
    return cleaned_df

if __name__ == "__main__":
    data = load_data("sample_data.csv")
    
    if data is not None:
        numeric_cols = ['age', 'income', 'score']
        cleaned_data = clean_data(data, numeric_cols)
        
        cleaned_data.to_csv("cleaned_data.csv", index=False)
        print("Cleaned data saved to 'cleaned_data.csv'")
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is False.
        fill_value: Value to use for filling missing values. Default is 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection. Default is 1.5.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed from specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1] using min-max scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to normalize.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_df = df.copy()
    min_val = normalized_df[column].min()
    max_val = normalized_df[column].max()
    
    if max_val != min_val:
        normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
    else:
        normalized_df[column] = 0
    
    return normalized_df