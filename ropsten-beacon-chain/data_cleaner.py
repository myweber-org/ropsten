import numpy as np
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
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def z_score_normalize(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_zscore'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr'):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = z_score_normalize(cleaned_df, col)
        elif method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return Trueimport pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Clean missing data in a CSV file using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file
        strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to clean, if None cleans all numeric columns
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns = list(numeric_cols)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
        output_path (str): Path to save the cleaned data
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_missing_data(input_file, strategy='median')
    
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
        print("Missing values after cleaning:")
        print(cleaned_df.isnull().sum())import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, threshold=1.5):
    """Detect outliers using the Interquartile Range method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data < lower_bound) | (data > upper_bound)

def remove_outliers(df, columns, threshold=1.5):
    """Remove outliers from specified columns in a DataFrame."""
    df_clean = df.copy()
    for col in columns:
        outliers = detect_outliers_iqr(df[col], threshold)
        df_clean = df_clean[~outliers]
    return df_clean.reset_index(drop=True)

def normalize_minmax(data):
    """Normalize data to [0, 1] range using min-max scaling."""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def standardize_zscore(data):
    """Standardize data using z-score normalization."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return np.zeros_like(data)
    return (data - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalization='standardize'):
    """Complete data cleaning pipeline."""
    df_clean = remove_outliers(df, numeric_columns, outlier_threshold)
    
    for col in numeric_columns:
        if normalization == 'minmax':
            df_clean[col] = normalize_minmax(df_clean[col].values)
        elif normalization == 'standardize':
            df_clean[col] = standardize_zscore(df_clean[col].values)
    
    return df_clean

def validate_data(df, required_columns, numeric_columns):
    """Validate dataset structure and content."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for col in numeric_columns:
        if col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                raise TypeError(f"Column {col} must be numeric")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.exponential(1, 100),
        'feature3': np.random.randint(1, 100, 100)
    })
    
    numeric_cols = ['feature1', 'feature2', 'feature3']
    required_cols = ['feature1', 'feature2', 'feature3']
    
    try:
        validate_data(sample_data, required_cols, numeric_cols)
        cleaned_data = clean_dataset(sample_data, numeric_cols, normalization='standardize')
        print(f"Original shape: {sample_data.shape}")
        print(f"Cleaned shape: {cleaned_data.shape}")
        print("Data cleaning completed successfully")
    except Exception as e:
        print(f"Data cleaning failed: {e}")