
import pandas as pd
import numpy as np
from scipy import stats

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

def clean_dataset(filepath, columns_to_clean):
    df = pd.read_csv(filepath)
    
    for column in columns_to_clean:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    cleaned_filepath = filepath.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_filepath, index=False)
    return cleaned_filepath

if __name__ == "__main__":
    data_file = "raw_data.csv"
    columns = ["temperature", "pressure", "humidity"]
    result = clean_dataset(data_file, columns)
    print(f"Cleaned data saved to: {result}")
import numpy as np
import pandas as pd

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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'values': np.random.normal(100, 15, 1000).tolist() + [500, -200]
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    print(calculate_summary_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    print(calculate_summary_stats(cleaned_df, 'values'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self.df
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self.df
    
    def standardize_columns(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std > 0:
                self.df[col] = (self.df[col] - mean) / std
        
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()

def clean_dataset(df, missing_strategy='mean', remove_outliers=True):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if remove_outliers:
        cleaner.remove_outliers_iqr()
    
    cleaner.standardize_columns()
    return cleaner.get_cleaned_data()
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text, keeping only alphanumeric and spaces by default.
    
    Args:
        text (str): Input text
        keep_pattern (str): Regex pattern of characters to keep
    
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    return re.sub(f'[^{keep_pattern}]', '', text)

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email (str): Email address to validate
    
    Returns:
        bool: True if email format is valid
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email).strip()))

def standardize_phone_number(phone):
    """
    Standardize phone number format to digits only.
    
    Args:
        phone (str): Phone number to standardize
    
    Returns:
        str: Phone number with only digits
    """
    if pd.isna(phone):
        return phone
    
    phone_str = str(phone)
    digits = re.sub(r'\D', '', phone_str)
    
    if len(digits) >= 10:
        return digits[-10:]
    
    return digits
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # If specific columns are provided, clean only those; otherwise clean all object columns
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns
    
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(_normalize_string)
    
    return df_clean, removed_duplicates

def _normalize_string(text):
    """
    Normalize a string: lowercase, strip whitespace, and remove extra spaces.
    """
    if pd.isna(text):
        return text
    text = str(text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and return a boolean series.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].apply(lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False)
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters except alphanumeric and spaces.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.replace(r'\s+', ' ', regex=True)
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def clean_dataset(df, text_columns=None, deduplicate=True):
    """
    Main function to clean dataset by processing text columns and removing duplicates.
    """
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if deduplicate:
        df = remove_duplicates(df)
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'JANE SMITH', 'John Doe  ', 'Alice & Bob'],
        'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'alice@example.com'],
        'notes': ['Important client', 'Regular customer', 'Important client', 'New contact']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, text_columns=['name', 'notes'])
    print(cleaned_df)import pandas as pd
import numpy as np

def clean_csv_data(filepath, strategy='mean', columns=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        strategy (str): Strategy for filling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop'.
        columns (list): List of column names to apply cleaning to. 
                       If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df = df.dropna(subset=[col])
        elif strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Missing values handled using '{strategy}' strategy")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        output_path (str): Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, strategy='median')
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")