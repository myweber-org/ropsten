import numpy as np
import pandas as pd

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize a column using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize a column using Z-score normalization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns using different strategies.
    """
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'constant'")
            
            data_filled[col] = data[col].fillna(fill_value)
    
    return data_filled

def create_sample_data():
    """
    Create sample data for testing the cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    # Add some outliers
    data['value'][0] = 200
    data['value'][1] = 30
    
    # Add some missing values
    data['value'][5] = np.nan
    data['value'][10] = np.nan
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Example usage
    df = create_sample_data()
    print("Original data shape:", df.shape)
    
    # Detect outliers
    outliers = detect_outliers_iqr(df, 'value')
    print(f"Detected {len(outliers)} outliers")
    
    # Remove outliers
    cleaned_df = remove_outliers(df, 'value')
    print("Data shape after removing outliers:", cleaned_df.shape)
    
    # Handle missing values
    filled_df = handle_missing_values(df, strategy='mean', columns=['value'])
    print("Missing values handled")
    
    # Normalize data
    df['value_normalized'] = normalize_minmax(df, 'value')
    df['value_standardized'] = standardize_zscore(df, 'value')
    
    print("\nSample of processed data:")
    print(df[['value', 'value_normalized', 'value_standardized']].head())