
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    date_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and converting data types.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
        date_columns: List of column names to parse as dates
    
    Returns:
        Cleaned DataFrame
    """
    
    # Read CSV file
    df = pd.read_csv(input_path)
    
    # Parse date columns if specified
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Handle missing values based on strategy
    if missing_strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == 'zero':
        df = df.fillna(0)
    elif missing_strategy == 'drop':
        df = df.dropna()
    
    # Convert object columns to categorical if they have few unique values
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 10:
            df[col] = df[col].astype('category')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['category']).columns),
        'date_columns': list(df.select_dtypes(include=['datetime']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    cleaned_df = clean_csv_data(
        input_path='raw_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean',
        date_columns=['transaction_date', 'created_at']
    )
    
    validation = validate_dataframe(cleaned_df)
    print(f"Data cleaning completed. Validation results: {validation}")