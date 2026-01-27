import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): Specific columns to apply cleaning. If None, applies to all.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if col in df_clean.columns:
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean.dropna(subset=[col], inplace=True)
            elif strategy == 'fill_zero':
                df_clean[col].fillna(0, inplace=True)
    
    return df_clean

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.Series: Boolean mask of outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    df_norm = df.copy()
    
    if method == 'minmax':
        min_val = df_norm[column].min()
        max_val = df_norm[column].max()
        if max_val != min_val:
            df_norm[column] = (df_norm[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_norm[column].mean()
        std_val = df_norm[column].std()
        if std_val != 0:
            df_norm[column] = (df_norm[column] - mean_val) / std_val
    
    return df_norm

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def save_cleaned_data(df, filename, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Output filename
        format (str): File format ('csv', 'excel', 'json')
    """
    if format == 'csv':
        df.to_csv(filename, index=False)
    elif format == 'excel':
        df.to_excel(filename, index=False)
    elif format == 'json':
        df.to_json(filename, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
import pandas as pd
import numpy as np
from pathlib import Path

def load_dataset(file_path):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from the dataframe."""
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep='first')
    removed_count = initial_count - len(df_clean)
    print(f"Removed {removed_count} duplicate rows.")
    return df_clean

def standardize_column(df, column_name, case='lower'):
    """Standardize text in a specific column."""
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in dataframe.")
        return df
    
    if case == 'lower':
        df[column_name] = df[column_name].astype(str).str.lower()
    elif case == 'upper':
        df[column_name] = df[column_name].astype(str).str.upper()
    elif case == 'title':
        df[column_name] = df[column_name].astype(str).str.title()
    
    print(f"Standardized column '{column_name}' to {case} case.")
    return df

def fill_missing_values(df, column_name, fill_value=np.nan):
    """Fill missing values in a column with specified value."""
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in dataframe.")
        return df
    
    missing_count = df[column_name].isnull().sum()
    df[column_name] = df[column_name].fillna(fill_value)
    print(f"Filled {missing_count} missing values in column '{column_name}'.")
    return df

def clean_dataset(input_path, output_path=None):
    """Main function to clean the dataset."""
    df = load_dataset(input_path)
    if df is None:
        return
    
    print("Starting data cleaning process...")
    
    df = remove_duplicates(df)
    df = standardize_column(df, 'name', case='title')
    df = standardize_column(df, 'email', case='lower')
    df = fill_missing_values(df, 'age', fill_value=0)
    df = fill_missing_values(df, 'salary', fill_value=df['salary'].median())
    
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Final dataset shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    clean_dataset(input_file)
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
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.array([300, 350, -50, -100])
        ])
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:", calculate_summary_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:", calculate_summary_stats(cleaned_df, 'values'))

if __name__ == "__main__":
    main()