import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """Load CSV data and perform cleaning operations."""
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))
    
    # Remove outliers using z-score method
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep: 'first', 'last', or False
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep, ignore_index=True)
    
    print(f"Removed {len(df) - len(cleaned_df)} duplicate rows")
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of column to clean
        fill_method (str): Method to fill missing values: 'mean', 'median', 'zero'
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame")
        return df
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Error: Column '{column_name}' is not numeric")
        return df
    
    missing_count = df[column_name].isna().sum()
    
    if missing_count > 0:
        if fill_method == 'mean':
            fill_value = df[column_name].mean()
        elif fill_method == 'median':
            fill_value = df[column_name].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            fill_value = df[column_name].mean()
        
        df[column_name] = df[column_name].fillna(fill_value)
        print(f"Filled {missing_count} missing values in '{column_name}' with {fill_method}: {fill_value}")
    
    return df

def get_data_summary(df):
    """
    Generate summary statistics for DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 35, 28, 28, 40],
        'score': [85.5, 92.0, 78.5, 88.0, 95.5, 95.5, 82.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_duplicates(df)
    print()
    
    cleaned_df = clean_numeric_column(cleaned_df, 'age', fill_method='mean')
    print()
    
    summary = get_data_summary(cleaned_df)
    print("Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")