
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data_frame):
        self.df = data_frame.copy()
        self.original_shape = data_frame.shape
        
    def remove_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = 0
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicate rows.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def filter_by_threshold(df, column, threshold):
    """
    Filter DataFrame rows where column value is greater than threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to apply filter on.
        threshold (float): Threshold value.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    filtered_df = df[df[column] > threshold]
    return filtered_df

def main():
    # Example usage
    data = {
        'A': [1, 2, None, 4, 4],
        'B': [5, None, 7, 8, 8],
        'C': [10, 20, 30, 40, 40]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    filtered_df = filter_by_threshold(cleaned_df, 'C', 25)
    print("Filtered DataFrame (C > 25):")
    print(filtered_df)

if __name__ == "__main__":
    main()import pandas as pd

def clean_dataframe(df, column_mapping=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by standardizing column names and removing duplicates.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Dictionary mapping original column names to standardized names
        remove_duplicates: Boolean indicating whether to remove duplicate rows
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Standardize column names
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Convert column names to lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame contains required columns and has no null values in key columns.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        return False, f"Completely empty columns: {empty_columns}"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Product Name': ['A', 'B', 'A', 'C', 'B'],
        'Price': [100, 200, 100, 300, 200],
        'Quantity': [5, 3, 5, 2, 3]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df)
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, ['product_name', 'price'])
    print(f"\nValidation: {is_valid}, Message: {message}")
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize=False):
    """
    Main function to clean dataset with multiple options
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        # Handle missing values
        if cleaned_df[col].isna().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Remove outliers
        if outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        elif outlier_method == 'iqr':
            outliers, _, _ = detect_outliers_iqr(cleaned_df, col)
            cleaned_df = cleaned_df[~cleaned_df.index.isin(outliers.index)]
        
        # Normalize if requested
        if normalize:
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[f'{col}_standardized'] = standardize_data(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_columns': [],
        'all_nan_columns': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    for col in df.columns:
        if df[col].empty:
            validation_results['empty_columns'].append(col)
            validation_results['is_valid'] = False
        elif df[col].isna().all():
            validation_results['all_nan_columns'].append(col)
            validation_results['is_valid'] = False
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    })
    
    print("Original Data:")
    print(sample_data)
    
    cleaned = clean_dataset(sample_data, numeric_columns=['A', 'B', 'C'], normalize=True)
    print("\nCleaned Data:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print("\nValidation Results:")
    print(validation)