
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    """Main data cleaning function."""
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ["age", "salary", "score"]
    
    raw_data = load_dataset(input_file)
    cleaned_data = clean_data(raw_data, numeric_cols)
    save_cleaned_data(cleaned_data, output_file)
    
    print(f"Original data shape: {raw_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Data cleaning completed. Saved to {output_file}")import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean'):
    """
    Load a CSV file and clean missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        fill_strategy (str): Strategy to fill missing values.
            Options: 'mean', 'median', 'zero', 'drop'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if df.empty:
        return df
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'drop':
        df_cleaned = df.dropna(subset=numeric_columns)
    else:
        df_cleaned = df.copy()
        for col in numeric_columns:
            if df_cleaned[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = df_cleaned[col].mean()
                elif fill_strategy == 'median':
                    fill_value = df_cleaned[col].median()
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown fill strategy: {fill_strategy}")
                df_cleaned[col].fillna(fill_value, inplace=True)
    
    return df_cleaned

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        column (str): Column name to check for outliers.
        threshold (float): IQR multiplier threshold.
    
    Returns:
        pandas.Series: Boolean series indicating outliers.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame.
        output_path (str): Path to save the cleaned CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")