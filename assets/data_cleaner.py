
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def clean_data(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    return df.dropna()

def main():
    data = load_data('raw_data.csv')
    numeric_cols = ['age', 'salary', 'score']
    cleaned_data = clean_data(data, numeric_cols)
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Original: {len(data)} rows, Cleaned: {len(cleaned_data)} rows")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def remove_outliers_zscore(dataframe, column, threshold=3):
    z_scores = np.abs(stats.zscore(dataframe[column]))
    return dataframe[z_scores < threshold]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    dataframe[column + '_normalized'] = (dataframe[column] - min_val) / (max_val - min_val)
    return dataframe

def normalize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    dataframe[column + '_standardized'] = (dataframe[column] - mean_val) / std_val
    return dataframe

def handle_missing_values(dataframe, strategy='mean'):
    if strategy == 'mean':
        return dataframe.fillna(dataframe.mean())
    elif strategy == 'median':
        return dataframe.fillna(dataframe.median())
    elif strategy == 'mode':
        return dataframe.fillna(dataframe.mode().iloc[0])
    elif strategy == 'drop':
        return dataframe.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(dataframe, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    df_clean = dataframe.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            df_clean = remove_outliers_iqr(df_clean, col)
        elif outlier_method == 'zscore':
            df_clean = remove_outliers_zscore(df_clean, col)
        
        if normalize_method == 'minmax':
            df_clean = normalize_minmax(df_clean, col)
        elif normalize_method == 'zscore':
            df_clean = normalize_zscore(df_clean, col)
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    return df_clean
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_na=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): Whether to remove duplicate rows.
    fill_na (optional): Value to fill missing entries with.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if fill_na is not None:
        cleaned_df = cleaned_df.fillna(fill_na)
    else:
        cleaned_df = cleaned_df.dropna()
    
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
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