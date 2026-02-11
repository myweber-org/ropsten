
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
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

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    numeric_cols = ['feature_a', 'feature_b']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(result.head())
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

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_minmax(dataframe, columns):
    """
    Normalize specified columns using min-max scaling
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def standardize_zscore(dataframe, columns):
    """
    Standardize specified columns using z-score normalization
    """
    df_standardized = dataframe.copy()
    for col in columns:
        if col in df_standardized.columns:
            mean_val = df_standardized[col].mean()
            std_val = df_standardized[col].std()
            if std_val > 0:
                df_standardized[col] = (df_standardized[col] - mean_val) / std_val
    return df_standardized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    df_processed = dataframe.copy()
    
    if columns is None:
        columns = df_processed.columns
    
    for col in columns:
        if col in df_processed.columns and df_processed[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_processed[col].mean()
            elif strategy == 'median':
                fill_value = df_processed[col].median()
            elif strategy == 'mode':
                fill_value = df_processed[col].mode()[0]
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with skewness above threshold
    """
    skewed_cols = []
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = abs(dataframe[col].skew())
        if skewness > threshold:
            skewed_cols.append((col, skewness))
    
    return sorted(skewed_cols, key=lambda x: x[1], reverse=True)

def apply_log_transform(dataframe, columns):
    """
    Apply log transformation to specified columns
    """
    df_transformed = dataframe.copy()
    for col in columns:
        if col in df_transformed.columns:
            # Add small constant to handle zero values
            df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed