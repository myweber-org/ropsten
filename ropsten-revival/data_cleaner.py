
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (np.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.ndarray: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Initial shape: {df.shape}")
    
    df_cleaned = df.copy()
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
        outliers = z_scores > 3
        df_cleaned.loc[outliers, col] = np.nan
        print(f"Removed {outliers.sum()} outliers from {col}")
    
    df_cleaned = df_cleaned.dropna()
    print(f"Shape after outlier removal: {df_cleaned.shape}")
    
    for col in numeric_cols:
        if col in df_cleaned.columns:
            col_min = df_cleaned[col].min()
            col_max = df_cleaned[col].max()
            if col_max > col_min:
                df_cleaned[col] = (df_cleaned[col] - col_min) / (col_max - col_min)
    
    print(f"Final cleaned shape: {df_cleaned.shape}")
    return df_cleaned

if __name__ == "__main__":
    cleaned_df = load_and_clean_data("sample_data.csv")
    cleaned_df.to_csv("cleaned_data.csv", index=False)
    print("Data cleaning complete. Saved to cleaned_data.csv")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    return stats

def process_numerical_data(data, numerical_columns):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    numerical_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Statistics for each processed column
    """
    cleaned_data = data.copy()
    all_stats = {}
    
    for col in numerical_columns:
        if col in cleaned_data.columns:
            original_count = len(cleaned_data)
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
            removed_count = original_count - len(cleaned_data)
            
            stats = calculate_summary_stats(cleaned_data, col)
            stats['outliers_removed'] = removed_count
            all_stats[col] = stats
    
    return cleaned_data, all_stats
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
        
        print("Filled missing values in numeric and categorical columns")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    """
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty")
    
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_percentage[missing_percentage > 50].index.tolist()
    
    if high_missing:
        issues.append(f"Columns with >50% missing values: {high_missing}")
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows")
    
    return issues

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10, 20, 20, None, 40, None],
        'category': ['A', 'B', 'B', None, 'C', 'A']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation issues:", validate_dataframe(df))
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nValidation issues after cleaning:", validate_dataframe(cleaned_df))

if __name__ == "__main__":
    main()