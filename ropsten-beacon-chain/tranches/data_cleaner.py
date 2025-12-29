import re

def normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping non-alphanumeric characters (except basic punctuation).
    """
    if not isinstance(text, str):
        return text

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Keep only alphanumeric characters, spaces, and basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    return text.strip()

def clean_dataframe_column(df, column_name):
    """
    Apply normalize_string to a specific column in a pandas DataFrame.
    """
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame) and column_name in df.columns:
            df[column_name] = df[column_name].apply(normalize_string)
        return df
    except ImportError:
        print("Pandas is not installed. Please install pandas to use this function.")
        return df
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
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

def process_dataset(file_path, column_to_clean):
    """
    Main function to load and clean a dataset.
    
    Args:
        file_path (str): Path to CSV file
        column_to_clean (str): Column name to clean
    
    Returns:
        tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_summary_statistics(df, column_to_clean)
    cleaned_df = remove_outliers_iqr(df, column_to_clean)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column_to_clean)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 15, 12, 11, 10, 9, 8, 12, 13, 14, 200]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned data:")
    print(cleaned_df)
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    print("\nSummary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    """Normalize specified column to range [0, 1]."""
    if column_name in df.columns:
        col_min = df[column_name].min()
        col_max = df[column_name].max()
        if col_max != col_min:
            df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
    return df

def clean_dataset(file_path, output_path=None):
    """Main function to clean dataset with default operations."""
    try:
        df = pd.read_csv(file_path)
        df = remove_duplicates(df)
        df = fill_missing_values(df, 'mean')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = normalize_column(df, col)
        
        if output_path:
            df.to_csv(output_path, index=False)
            return f"Cleaned data saved to {output_path}"
        return df
    except Exception as e:
        return f"Error cleaning dataset: {str(e)}"