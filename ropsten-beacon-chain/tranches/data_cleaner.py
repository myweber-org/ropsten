import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, method='standardize'):
    """
    Main cleaning function for datasets
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            
            # Apply normalization/standardization
            if method == 'normalize':
                cleaned_data[column] = normalize_minmax(cleaned_data, column)
            elif method == 'standardize':
                cleaned_data[column] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(data) < min_rows:
        raise ValueError(f"Dataset has only {len(data)} rows, minimum required is {min_rows}")
    
    if data.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values")
    
    return True

def example_usage():
    """
    Example usage of the data cleaning utilities
    """
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some outliers
    sample_data.loc[0, 'feature_a'] = 500
    sample_data.loc[1, 'feature_b'] = 1000
    
    print("Original data shape:", sample_data.shape)
    print("Original data stats:")
    print(sample_data[['feature_a', 'feature_b']].describe())
    
    # Clean the data
    cleaned = clean_dataset(
        sample_data,
        numeric_columns=['feature_a', 'feature_b'],
        outlier_factor=1.5,
        method='standardize'
    )
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned data stats:")
    print(cleaned[['feature_a', 'feature_b']].describe())
    
    # Validate the cleaned data
    try:
        validate_data(cleaned, ['feature_a', 'feature_b', 'category'], min_rows=50)
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")
    
    return cleaned

if __name__ == "__main__":
    cleaned_data = example_usage()