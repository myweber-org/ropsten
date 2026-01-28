
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to process
        factor: multiplier for IQR (default 1.5)
    
    Returns:
        Cleaned DataFrame with outliers removed
    """
    df_clean = dataframe.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(dataframe, columns):
    """
    Normalize data using Min-Max scaling to [0, 1] range.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    df_norm = dataframe.copy()
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        
        if max_val != min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    
    return df_norm

def standardize_zscore(dataframe, columns):
    """
    Standardize data using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    df_std = dataframe.copy()
    
    for col in columns:
        if col not in df_std.columns:
            continue
            
        mean_val = df_std[col].mean()
        std_val = df_std[col].std()
        
        if std_val > 0:
            df_std[col] = (df_std[col] - mean_val) / std_val
        else:
            df_std[col] = 0
    
    return df_std

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    df_processed = dataframe.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        df_processed = df_processed.dropna(subset=columns)
    else:
        for col in columns:
            if col not in df_processed.columns:
                continue
                
            if strategy == 'mean':
                fill_value = df_processed[col].mean()
            elif strategy == 'median':
                fill_value = df_processed[col].median()
            elif strategy == 'mode':
                fill_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 0
            else:
                fill_value = 0
            
            df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed.reset_index(drop=True)

def clean_dataset(dataframe, numeric_columns=None, outlier_factor=1.5, 
                  normalize=False, standardize=False, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_factor: IQR factor for outlier removal
        normalize: whether to apply min-max normalization
        standardize: whether to apply z-score standardization
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned and processed DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = dataframe.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy, columns=numeric_columns)
    df_clean = remove_outliers_iqr(df_clean, columns=numeric_columns, factor=outlier_factor)
    
    if normalize:
        df_clean = normalize_minmax(df_clean, columns=numeric_columns)
    elif standardize:
        df_clean = standardize_zscore(df_clean, columns=numeric_columns)
    
    return df_clean

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(cleaned_df[col]))
            cleaned_df = cleaned_df[z_scores < 3]
    
    for col in numeric_columns:
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_cleaning(original_df, cleaned_df):
    report = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100
    }
    return report
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary mapping original column names to new names
        drop_duplicates: Boolean flag to remove duplicate rows
        normalize_text: Boolean flag to normalize text columns
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    normalized = str(text).lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df: Input pandas DataFrame
        email_column: Name of the column containing email addresses
    
    Returns:
        DataFrame with validation results
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validation_results = df.copy()
    validation_results['email_valid'] = validation_results[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = validation_results['email_valid'].sum()
    total_count = len(validation_results)
    
    print(f"Email validation: {valid_count}/{total_count} valid emails ({valid_count/total_count*100:.1f}%)")
    
    return validation_results