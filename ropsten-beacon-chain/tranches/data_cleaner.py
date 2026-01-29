
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def clean_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): Strategy for imputation ('mean', 'median', 'drop').
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return df

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1] using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to normalize.
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        df[column] = 0.5
    else:
        df[column] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def process_dataset(file_path, output_path=None):
    """
    Complete data cleaning pipeline for a CSV file.
    
    Parameters:
    file_path (str): Path to input CSV file.
    output_path (str, optional): Path to save cleaned data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(file_path)
    
    print(f"Original shape: {df.shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    df = clean_missing_values(df, strategy='median')
    
    for col in numeric_cols:
        if col in df.columns:
            df = normalize_column(df, col)
    
    print(f"Cleaned shape: {df.shape}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(50, 15, 100),
        'B': np.random.exponential(2, 100),
        'C': np.random.randint(1, 100, 100)
    })
    
    sample_data.loc[10:15, 'A'] = np.nan
    sample_data.loc[5, 'B'] = 1000
    
    cleaned = remove_outliers_iqr(sample_data, 'A')
    print(f"After outlier removal: {cleaned.shape}")
    
    normalized = normalize_column(cleaned.copy(), 'B')
    print("Normalization completed")