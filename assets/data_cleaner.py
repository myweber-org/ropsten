import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list, optional): Column labels to consider for duplicates.
    keep (str, optional): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to numeric and filling NaN.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
    return df

def validate_dataframe(df, required_columns):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'score': ['85', '90', '90', 'invalid', '88']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    print("\nAfter removing duplicates:")
    print(df_clean)
    
    df_clean = clean_numeric_columns(df_clean, ['score'])
    print("\nAfter cleaning numeric columns:")
    print(df_clean)
    
    is_valid = validate_dataframe(df_clean, ['id', 'name', 'score'])
    print(f"\nDataFrame validation: {is_valid}")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def calculate_statistics(df, column):
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    return stats

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    
    print("Original dataset shape:", sample_data.shape)
    print("Original statistics:")
    for col in sample_data.columns:
        print(f"{col}: {calculate_statistics(sample_data, col)}")
    
    cleaned_data = clean_dataset(sample_data, ['A', 'B', 'C'])
    
    print("\nCleaned dataset shape:", cleaned_data.shape)
    print("Cleaned statistics:")
    for col in cleaned_data.columns:
        print(f"{col}: {calculate_statistics(cleaned_data, col)}")