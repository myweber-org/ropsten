
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, columns, factor=1.5):
    """
    Remove outliers using IQR method
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

def remove_outliers_zscore(dataframe, columns, threshold=3):
    """
    Remove outliers using Z-score method
    """
    df_clean = dataframe.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
        mask = z_scores < threshold
        
        valid_indices = df_clean[col].dropna().index[mask]
        df_clean = df_clean.loc[valid_indices]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(dataframe, columns):
    """
    Normalize data using Min-Max scaling
    """
    df_normalized = dataframe.copy()
    
    for col in columns:
        if col not in df_normalized.columns:
            continue
            
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        
        if max_val != min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0
    
    return df_normalized

def normalize_zscore(dataframe, columns):
    """
    Normalize data using Z-score standardization
    """
    df_normalized = dataframe.copy()
    
    for col in columns:
        if col not in df_normalized.columns:
            continue
            
        mean_val = df_normalized[col].mean()
        std_val = df_normalized[col].std()
        
        if std_val != 0:
            df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        else:
            df_normalized[col] = 0
    
    return df_normalized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    df_clean = dataframe.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_clean[col].mean()
        elif strategy == 'median':
            fill_value = df_clean[col].median()
        elif strategy == 'mode':
            fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
            continue
        else:
            fill_value = 0
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean.reset_index(drop=True)

def get_data_summary(dataframe):
    """
    Generate comprehensive data summary
    """
    summary = {
        'shape': dataframe.shape,
        'dtypes': dataframe.dtypes.to_dict(),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': dataframe[col].mean(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max(),
            'median': dataframe[col].median(),
            'skewness': dataframe[col].skew()
        }
    
    return summary