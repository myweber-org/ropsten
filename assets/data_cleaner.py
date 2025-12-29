import numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val != min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    
    return df_norm

def standardize_zscore(df, columns):
    """
    Standardize data using Z-score normalization.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    df_std = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val > 0:
            df_std[col] = (df[col] - mean_val) / std_val
        else:
            df_std[col] = 0
    
    return df_std

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    else:
        for col in columns:
            if col not in df.columns:
                continue
                
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            else:
                fill_value = 0
            
            df_clean[col] = df[col].fillna(fill_value)
    
    return df_clean.reset_index(drop=True)

def create_data_summary(df):
    """
    Create a summary statistics DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    summary = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.values,
        'non_null': df.count().values,
        'null_count': df.isnull().sum().values,
        'null_percentage': (df.isnull().sum() / len(df) * 100).values,
        'unique_values': df.nunique().values
    })
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe().T
        summary = summary.merge(
            numeric_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']],
            left_on='column',
            right_index=True,
            how='left'
        )
    
    return summary

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_list = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned_list = remove_duplicates(sample_list)
    print(f"Original list: {sample_list}")
    print(f"List after removing duplicates: {cleaned_list}")