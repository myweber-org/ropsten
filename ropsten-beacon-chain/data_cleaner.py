
import pandas as pd

def clean_dataframe(df, column_to_deduplicate, columns_to_normalize=None):
    """
    Clean a pandas DataFrame by removing duplicates from a specified column
    and normalizing string columns to lowercase and stripping whitespace.
    """
    # Remove duplicates based on the specified column
    df_cleaned = df.drop_duplicates(subset=[column_to_deduplicate], keep='first')
    
    # Normalize specified string columns
    if columns_to_normalize:
        for col in columns_to_normalize:
            if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].astype(str).str.lower().str.strip()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    Returns a boolean and a list of missing columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'user_id': [1, 2, 3, 2, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'bob', 'David'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'bob@example.com', 'david@example.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    cleaned_df = clean_dataframe(df, 'user_id', columns_to_normalize=['name', 'email'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    # Validate the cleaned data
    required_cols = ['user_id', 'name', 'email']
    is_valid, missing = validate_dataframe(cleaned_df, required_cols)
    print(f"Data validation passed: {is_valid}")
    if not is_valid:
        print(f"Missing columns: {missing}")
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
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

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of removal statistics for each column
    """
    cleaned_df = df.copy()
    removal_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            removal_stats[column] = {
                'original_rows': original_count,
                'remaining_rows': len(cleaned_df),
                'removed_rows': removed_count,
                'removal_percentage': (removed_count / original_count) * 100
            }
    
    return cleaned_df, removal_stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000)
    })
    
    # Add some outliers
    sample_data.loc[50, 'temperature'] = 100
    sample_data.loc[150, 'humidity'] = 150
    sample_data.loc[250, 'pressure'] = 2000
    
    print("Original dataset shape:", sample_data.shape)
    
    columns_to_clean = ['temperature', 'humidity', 'pressure']
    cleaned_data, stats = clean_dataset(sample_data, columns_to_clean)
    
    print("\nCleaned dataset shape:", cleaned_data.shape)
    print("\nRemoval statistics:")
    for col, stat in stats.items():
        print(f"{col}: Removed {stat['removed_rows']} rows ({stat['removal_percentage']:.2f}%)")
    
    print("\nSummary statistics for cleaned data:")
    for column in columns_to_clean:
        col_stats = calculate_summary_statistics(cleaned_data, column)
        print(f"\n{column}:")
        for key, value in col_stats.items():
            print(f"  {key}: {value:.2f}")