
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, columns):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    df.loc[::90, 'B'] = 600
    
    print("Original shape:", df.shape)
    cleaned = clean_dataset(df, ['A', 'B'])
    print("Cleaned shape:", cleaned.shape)
    print("Outliers removed:", df.shape[0] - cleaned.shape[0])
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    columns_to_drop: Optional[list] = None
) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values and dropping specified columns.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path where cleaned CSV will be saved
    missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns_to_drop: List of column names to remove from dataset
    
    Returns:
    Cleaned pandas DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if missing_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif missing_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif missing_strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                df = df.dropna(subset=[col])
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
    print(f"Original shape: {pd.read_csv(input_path).shape}")
    print(f"Cleaned shape: {df.shape}")
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate dataframe for common data quality issues.
    
    Returns dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10.5, np.nan, 30.2, 40.1, 50.0],
        'C': ['X', 'Y', 'Z', np.nan, 'X'],
        'D': [100, 200, 300, 400, 500]
    })
    
    sample_df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean',
        columns_to_drop=['D']
    )
    
    validation = validate_dataframe(cleaned_df)
    print("\nData Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")