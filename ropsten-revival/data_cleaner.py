
import pandas as pd
import re

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, convert to lowercase).
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_names (list): List of column names to normalize
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize specified string columns
    for col in column_names:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using regex pattern.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with valid email flag column added
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].astype(str).str.match(email_pattern)
    
    return df

def remove_outliers_iqr(df, column_name):
    """
    Remove outliers from a numeric column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of numeric column to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    return df_filtered.reset_index(drop=True)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.reset_index(drop=True)

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
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
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(100),
        'value': np.random.randn(100) * 10 + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[10, 'value'] = 200
    sample_df.loc[20, 'value'] = -100
    
    print("Original dataset shape:", sample_df.shape)
    cleaned_df = clean_dataset(sample_df, ['value'])
    print("Cleaned dataset shape:", cleaned_df.shape)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list, optional): List of numeric columns to clean.
            If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    original_rows = len(cleaned_df)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    removed_rows = original_rows - len(cleaned_df)
    print(f"Removed {removed_rows} outliers from dataset")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 100, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 200, -5],
        'location': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nCleaned dataset:")
    cleaned = clean_dataset(df, ['temperature', 'humidity'])
    print(cleaned)
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers using IQR method")
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                else:
                    self.df[col] = 0
                    
        print("Applied Min-Max normalization")
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                else:
                    self.df[col] = 0
                    
        print("Applied Z-score standardization")
        return self
        
    def handle_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if self.df[col].dtype in [np.float64, np.int64]:
                    fill_value = self.df[col].median()
                    self.df[col].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in {col} with median: {fill_value}")
                    
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[95, 'feature_b'] = 1000
    df.loc[96, 'feature_b'] = -500
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("Missing values:", sample_df.isnull().sum().sum())
    
    cleaner = DataCleaner(sample_df)
    cleaned_df = (cleaner
                 .handle_missing_median()
                 .remove_outliers_iqr(['feature_a', 'feature_b'])
                 .standardize_zscore(['feature_a', 'feature_b'])
                 .normalize_minmax(['feature_c'])
                 .get_cleaned_data())
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    for column in numeric_columns:
        df = remove_outliers_iqr(df, column)
        df = normalize_column(df, column)
    return df

if __name__ == "__main__":
    sample_data = {
        'feature1': [10, 12, 14, 100, 15, 13, 11, 16, 12, 14],
        'feature2': [20, 22, 24, 200, 25, 23, 21, 26, 22, 24]
    }
    df = pd.DataFrame(sample_data)
    cleaned_df = clean_dataset(df, ['feature1', 'feature2'])
    print("Original dataset:")
    print(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    df_cleaned = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_minmax(df_cleaned, numeric_columns)
    return df_normalized

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print("Data cleaning completed. Saved to cleaned_data.csv")
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: dictionary mapping old column names to new ones
        drop_duplicates: whether to remove duplicate rows
        normalize_text: whether to normalize text columns (lowercase, strip whitespace)
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].astype(str).apply(
                lambda x: re.sub(r'\s+', ' ', x.strip().lower())
            )
    
    return df_clean

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email: string email address to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def extract_numeric(text):
    """
    Extract numeric values from text string.
    
    Args:
        text: string containing numeric values
    
    Returns:
        List of numeric values found in text
    """
    numbers = re.findall(r'\d+\.?\d*', text)
    return [float(num) if '.' in num else int(num) for num in numbers]
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data(data_list):
    if not isinstance(data_list, list):
        raise TypeError("Input must be a list")
    return remove_duplicates(data_list)