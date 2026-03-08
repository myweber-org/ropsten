
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def z_score_normalize(data, column):
    """
    Normalize data using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        min_target, max_target = feature_range
        normalized = normalized * (max_target - min_target) + min_target
    
    return normalized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
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
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
            
            data_clean[col] = data_clean[col].fillna(fill_value)
    
    return data_clean

def validate_dataframe(data, required_columns=None):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'feature1': np.random.normal(50, 15, n_samples),
        'feature2': np.random.exponential(2, n_samples),
        'feature3': np.random.randint(1, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    data.loc[np.random.choice(n_samples, 5, replace=False), 'feature1'] = np.nan
    data.loc[np.random.choice(n_samples, 3, replace=False), 'feature2'] = np.nan
    
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    
    cleaned_data, removed = remove_outliers_iqr(sample_data, 'feature1')
    print(f"After outlier removal: {cleaned_data.shape}, removed {removed} rows")
    
    normalized_data = cleaned_data.copy()
    normalized_data['feature1_norm'] = z_score_normalize(cleaned_data, 'feature1')
    normalized_data['feature2_norm'] = min_max_normalize(cleaned_data, 'feature2')
    
    filled_data = handle_missing_values(normalized_data, strategy='mean')
    print("Final cleaned data shape:", filled_data.shape)
    
    try:
        validate_dataframe(filled_data, ['feature1', 'feature2'])
        print("Data validation passed")
    except ValueError as e:
        print(f"Data validation failed: {e}")