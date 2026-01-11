
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def main():
    # Example usage
    data = {'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11]}
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal statistics:")
    print(calculate_summary_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned statistics:")
    print(calculate_summary_stats(cleaned_df, 'values'))

if __name__ == "__main__":
    main()
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: pandas DataFrame to clean
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to mark
            'first' : Mark duplicates as False except for the first occurrence.
            'last' : Mark duplicates as False except for the last occurrence.
            False : Mark all duplicates as True.
    
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Boolean indicating if validation passed
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def clean_dataset(file_path, output_path=None):
    """
    Main function to clean a dataset from file.
    
    Args:
        file_path: path to input data file
        output_path: path to save cleaned data (optional)
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
    print(f"Original dataset shape: {df.shape}")
    
    df = remove_duplicates(df)
    df = clean_numeric_columns(df, df.select_dtypes(include=['object']).columns)
    
    df = df.dropna()
    
    print(f"Cleaned dataset shape: {df.shape}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'age': ['25', '30', '30', '35', '40'],
        'score': ['85', '90', '90', '95', '100']
    })
    
    print("Sample data before cleaning:")
    print(sample_data)
    
    cleaned = remove_duplicates(sample_data, subset=['id', 'name'])
    cleaned = clean_numeric_columns(cleaned, ['age', 'score'])
    
    print("\nSample data after cleaning:")
    print(cleaned)