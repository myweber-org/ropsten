import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalization(data, column):
    """
    Normalize data using Z-score.
    """
    mean = data[column].mean()
    std = data[column].std()
    data[column + '_normalized'] = (data[column] - mean) / std
    return data

def min_max_normalization(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_scaled'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns):
    """
    Main cleaning pipeline for numeric columns.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = z_score_normalization(cleaned_df, col)
            cleaned_df = min_max_normalization(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000)
    })
    numeric_cols = ['feature1', 'feature2']
    result = clean_dataset(sample_data, numeric_cols)
    print(result.head())
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, columns, threshold=1.5):
    """
    Remove outliers using IQR method for specified columns.
    Returns cleaned dataframe and outlier indices.
    """
    cleaned_df = dataframe.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        col_outliers = cleaned_df[(cleaned_df[col] < lower_bound) | 
                                  (cleaned_df[col] > upper_bound)].index
        outlier_indices.extend(col_outliers)
        
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & 
                                (cleaned_df[col] <= upper_bound)]
    
    outlier_indices = list(set(outlier_indices))
    return cleaned_df, outlier_indices

def normalize_minmax(dataframe, columns):
    """
    Apply min-max normalization to specified columns.
    Returns dataframe with normalized columns.
    """
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max != col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0
    
    return normalized_df

def standardize_zscore(dataframe, columns):
    """
    Apply z-score standardization to specified columns.
    Returns dataframe with standardized columns.
    """
    standardized_df = dataframe.copy()
    
    for col in columns:
        if col not in standardized_df.columns:
            continue
            
        col_mean = standardized_df[col].mean()
        col_std = standardized_df[col].std()
        
        if col_std > 0:
            standardized_df[col] = (standardized_df[col] - col_mean) / col_std
        else:
            standardized_df[col] = 0
    
    return standardized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
        elif strategy == 'mean':
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strategy == 'median':
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strategy == 'mode':
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    
    return df_copy

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    Returns validation result and error message.
    """
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"