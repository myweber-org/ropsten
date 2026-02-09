import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Remove duplicate rows and standardize text in specified columns.
    """
    # Remove duplicates
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(standardize_text)
    
    return df_clean

def standardize_text(text):
    """
    Standardize text: lowercase, remove extra whitespace, and special characters.
    """
    if pd.isna(text):
        return text
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters except alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def validate_email(email):
    """
    Validate email format using regex.
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def get_data_summary(df):
    """
    Generate summary statistics for the dataframe.
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    return summary
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
        
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: Column name to analyze
        
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean a dataset by removing outliers from multiple numeric columns.
    
    Args:
        df: pandas DataFrame
        numeric_columns: List of column names to clean
        
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nOriginal statistics:")
    for col in df.columns:
        print(f"{col}: {calculate_statistics(df, col)}")
    
    cleaned_df = clean_dataset(df, ['temperature', 'humidity', 'pressure'])
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\nCleaned statistics:")
    for col in cleaned_df.columns:
        print(f"{col}: {calculate_statistics(cleaned_df, col)}")