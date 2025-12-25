
import numpy as np
import pandas as pd

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
    
    return filtered_df

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

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", sample_df.shape)
    print("Original statistics for column 'A':")
    print(calculate_summary_statistics(sample_df, 'A'))
    
    cleaned_df = clean_dataset(sample_df, ['A', 'B'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            initial_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = initial_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column '{col}'")
        
        cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_file_path, index=False)
        print(f"Cleaned data saved to: {cleaned_file_path}")
        return cleaned_file_path
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return None

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[::100, 'A'] = 500
    sample_df.loc[::50, 'B'] = 300
    
    sample_df.to_csv('sample_data.csv', index=False)
    clean_dataset('sample_data.csv')import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def remove_outliers_iqr(df, column):
    """Remove outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    return filtered_df

def normalize_column(df, column):
    """Normalize a column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        print(f"Warning: Column '{column}' has constant values. Skipping normalization.")
        return df
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    print(f"Normalized column '{column}'")
    return df

def clean_data(df, numeric_columns):
    """Main cleaning function."""
    if df is None or df.empty:
        print("No data to clean.")
        return df
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
        else:
            print(f"Warning: Column '{col}' not found in data.")
    
    for col in numeric_columns:
        if col in df.columns:
            df = normalize_column(df, col)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Rows removed: {original_shape[0] - df.shape[0]}")
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    data = load_data(input_file)
    if data is not None:
        cleaned_data = clean_data(data, numeric_cols)
        save_cleaned_data(cleaned_data, output_file)