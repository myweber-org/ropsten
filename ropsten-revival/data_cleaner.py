import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_method='iqr', fill_value=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'constant')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    fill_value: Value to use when strategy='constant'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif strategy == 'constant' and fill_value is not None:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    # Handle outliers
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif outlier_method == 'zscore':
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            threshold = 3
            
            # Replace outliers with median
            median_val = cleaned_df[col].median()
            cleaned_df.loc[z_scores > threshold, col] = median_val
    
    return cleaned_df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset: Columns to consider for duplicates
    keep: Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_data(df, method='minmax'):
    """
    Normalize numeric columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): Normalization method ('minmax', 'zscore')
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    normalized_df = df.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'minmax':
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned = clean_dataset(df, strategy='median', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Normalize the data
    normalized = normalize_data(cleaned, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized)import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing invalid columns.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
    drop_threshold (float): Drop columns if missing value ratio exceeds this threshold.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Drop columns with too many missing values
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=cols_to_drop)
    
    if len(cols_to_drop) > 0:
        print(f"Dropped columns due to high missing values: {list(cols_to_drop)}")
    
    # Fill missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        fill_values = df[numeric_cols].mean()
    elif fill_strategy == 'median':
        fill_values = df[numeric_cols].median()
    elif fill_strategy == 'mode':
        fill_values = df[numeric_cols].mode().iloc[0]
    elif fill_strategy == 'zero':
        fill_values = 0
    else:
        raise ValueError("Invalid fill_strategy. Choose from 'mean', 'median', 'mode', 'zero'")
    
    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    
    # For non-numeric columns, fill with the most frequent value
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        most_frequent = df[col].mode()[0] if not df[col].mode().empty else ''
        df[col] = df[col].fillna(most_frequent)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Total missing values after cleaning: {df.isnull().sum().sum()}")
    
    return df

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV.
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data exported to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.3)
        export_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")