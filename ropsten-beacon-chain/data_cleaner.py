import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def zscore_normalize(data, column):
    """
    Normalize data using z-score method
    """
    mean = data[column].mean()
    std = data[column].std()
    data[column + '_normalized'] = (data[column] - mean) / std
    return data

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    a, b = feature_range
    data[column + '_scaled'] = a + ((data[column] - min_val) * (b - a)) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='zscore'):
    """
    Main cleaning pipeline for numeric columns
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            
            if normalize_method == 'zscore':
                cleaned_df = zscore_normalize(cleaned_df, col)
            elif normalize_method == 'minmax':
                cleaned_df = minmax_normalize(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns):
    """
    Validate dataframe structure and content
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return Trueimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)
            df[col] = normalize_minmax(df, col)
        
        df.dropna(inplace=True)
        print(f"Cleaned shape: {df.shape}")
        
        output_path = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return None

if __name__ == "__main__":
    cleaned_file = clean_dataset("sample_data.csv")
    if cleaned_file:
        print(f"Cleaned data saved to: {cleaned_file}")