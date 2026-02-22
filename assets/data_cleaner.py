
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for missing values - 'mean', 'median', 'mode', or 'drop'
    outlier_method (str): Method for outlier detection - 'iqr' or 'zscore'
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].dtype in [np.float64, np.int64]:
            handle_missing_values(df_clean, col, missing_strategy)
            handle_outliers(df_clean, col, outlier_method)
    
    return df_clean

def handle_missing_values(df, column, strategy):
    """Handle missing values in a column using specified strategy."""
    if df[column].isnull().sum() == 0:
        return
    
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mode':
        fill_value = df[column].mode()[0]
    elif strategy == 'drop':
        df.dropna(subset=[column], inplace=True)
        return
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    df[column].fillna(fill_value, inplace=True)

def handle_outliers(df, column, method):
    """Detect and handle outliers in a column."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        df.loc[~mask, column] = np.nan
        
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        z_scores = np.abs((df[column] - mean_val) / std_val)
        
        mask = z_scores <= 3
        df.loc[~mask, column] = np.nan
        
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    
    df[column].fillna(df[column].median(), inplace=True)

def get_cleaning_summary(df_original, df_cleaned):
    """Generate a summary of cleaning operations performed."""
    summary = {
        'original_rows': len(df_original),
        'cleaned_rows': len(df_cleaned),
        'rows_removed': len(df_original) - len(df_cleaned),
        'columns_cleaned': list(df_cleaned.select_dtypes(include=[np.number]).columns)
    }
    
    for col in summary['columns_cleaned']:
        if col in df_original.columns:
            original_missing = df_original[col].isnull().sum()
            cleaned_missing = df_cleaned[col].isnull().sum()
            summary[f'{col}_missing_fixed'] = original_missing - cleaned_missing
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14],
        'category': ['X', 'Y', 'X', 'Y', 'Z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    df_cleaned = clean_dataset(df, missing_strategy='median', outlier_method='iqr')
    print("Cleaned DataFrame:")
    print(df_cleaned)
    print("\n")
    
    summary = get_cleaning_summary(df, df_cleaned)
    print("Cleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")