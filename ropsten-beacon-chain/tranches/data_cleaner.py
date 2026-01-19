import numpy as np
import pandas as pd

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return cleaned_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def handle_missing_values(data, strategy='mean'):
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    elif strategy == 'drop':
        return data.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(data, numeric_columns, outlier_handling='remove', normalization='minmax', missing_strategy='mean'):
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            if outlier_handling == 'remove':
                cleaned_data = remove_outliers(cleaned_data, col)
            elif outlier_handling == 'mark':
                outliers = detect_outliers_iqr(cleaned_data, col)
                cleaned_data[col + '_is_outlier'] = cleaned_data.index.isin(outliers.index)
            
            if normalization == 'minmax':
                cleaned_data = normalize_minmax(cleaned_data, col)
            elif normalization == 'zscore':
                cleaned_data = standardize_zscore(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    return cleaned_data

def validate_data(data, required_columns, numeric_ranges=None):
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if numeric_ranges:
        for col, (min_val, max_val) in numeric_ranges.items():
            if col in data.columns:
                invalid_values = data[(data[col] < min_val) | (data[col] > max_val)]
                if not invalid_values.empty:
                    print(f"Warning: Column '{col}' has values outside range [{min_val}, {max_val}]")
    
    return True

def example_usage():
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'age': np.random.normal(35, 10, 100),
        'income': np.random.normal(50000, 15000, 100),
        'score': np.random.uniform(0, 100, 100)
    })
    
    sample_data.loc[10, 'age'] = 150
    sample_data.loc[20, 'income'] = 200000
    sample_data.loc[5:10, 'score'] = np.nan
    
    print("Original data shape:", sample_data.shape)
    print("Missing values:\n", sample_data.isnull().sum())
    
    cleaned = clean_dataset(
        sample_data,
        numeric_columns=['age', 'income', 'score'],
        outlier_handling='remove',
        normalization='zscore',
        missing_strategy='mean'
    )
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned data columns:", cleaned.columns.tolist())
    
    try:
        validate_data(cleaned, required_columns=['age', 'income', 'score'])
        print("Data validation passed")
    except ValueError as e:
        print(f"Data validation failed: {e}")
    
    return cleaned

if __name__ == "__main__":
    result = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result.head())
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print(f"Original dataset shape: {df.shape}")
    cleaned_df = clean_dataset(df)
    print(f"Cleaned dataset shape: {cleaned_df.shape}")