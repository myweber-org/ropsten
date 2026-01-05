import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_method (str or None): Method to fill missing values.
            Options: 'ffill', 'bfill', 'mean', 'median', or None to drop rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is None:
        cleaned_df = cleaned_df.dropna()
    elif fill_method in ['ffill', 'bfill']:
        cleaned_df = cleaned_df.fillna(method=fill_method)
    elif fill_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, message)
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

# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    sample_data = {
        'A': [1, 2, None, 4, 4],
        'B': [5, None, 7, 8, 8],
        'C': ['x', 'y', 'z', 'x', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)
    
    # Validate
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd

def clean_dataset(df, text_columns=None):
    """
    Clean a pandas DataFrame by removing rows with null values
    and standardizing text columns to lowercase.
    """
    df_clean = df.copy()
    
    df_clean = df_clean.dropna()
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
    
    return df_clean

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def validate_email_column(df, email_col):
    """
    Validate email addresses in a column and return valid rows.
    """
    if email_col not in df.columns:
        raise ValueError(f"Column '{email_col}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    valid_emails = df[df[email_col].str.match(email_pattern, na=False)]
    
    return valid_emails
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)
                clean_data = clean_data[mask]
                
        return clean_data.reset_index(drop=True)
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                z_scores = np.abs(stats.zscore(clean_data[col].dropna()))
                mask = z_scores < threshold
                clean_data = clean_data[mask]
                
        return clean_data.reset_index(drop=True)
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(normalized_data[col]):
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                
                if max_val != min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
                else:
                    normalized_data[col] = 0
                    
        return normalized_data
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(normalized_data[col]):
                mean_val = normalized_data[col].mean()
                std_val = normalized_data[col].std()
                
                if std_val > 0:
                    normalized_data[col] = (normalized_data[col] - mean_val) / std_val
                else:
                    normalized_data[col] = 0
                    
        return normalized_data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        filled_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(filled_data[col]):
                if strategy == 'mean':
                    fill_value = filled_data[col].mean()
                elif strategy == 'median':
                    fill_value = filled_data[col].median()
                elif strategy == 'mode':
                    fill_value = filled_data[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                    
                filled_data[col] = filled_data[col].fillna(fill_value)
                
        return filled_data
    
    def get_cleaning_report(self):
        report = {
            'original_shape': self.original_shape,
            'current_shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(exclude=[np.number]).columns)
        }
        return report

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    data['feature_a'][np.random.choice(1000, 50)] = np.nan
    data['feature_b'][np.random.choice(1000, 30)] = np.nan
    
    outliers = np.random.choice(1000, 20, replace=False)
    data['feature_a'][outliers] = data['feature_a'][outliers] * 5
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_shape)
    print("\nMissing values:")
    print(sample_df.isnull().sum())
    
    cleaned_df = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print("\nAfter IQR outlier removal:", cleaned_df.shape)
    
    normalized_df = cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print("\nAfter min-max normalization:")
    print(normalized_df[['feature_a', 'feature_b', 'feature_c']].describe())
    
    report = cleaner.get_cleaning_report()
    print("\nCleaning report:")
    for key, value in report.items():
        print(f"{key}: {value}")