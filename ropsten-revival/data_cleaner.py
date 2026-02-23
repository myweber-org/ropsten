import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Clean a DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        text_columns (list, optional): List of column names to standardize text.
            If None, all object dtype columns are processed.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Identify text columns if not specified
    if text_columns is None:
        text_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Standardize text in specified columns
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(_standardize_text)
    
    return df_clean

def _standardize_text(text):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text: Input text (can be any type).
    
    Returns:
        str: Standardized text or original value if not a string.
    """
    if not isinstance(text, str):
        return text
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                self.df[col].fillna(fill_value, inplace=True)
        
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        self.df = df_clean
        return self

    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in df_normalized.columns:
                if method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = df_normalized[col].mean()
                    std_val = df_normalized[col].std()
                    if std_val != 0:
                        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
                else:
                    raise ValueError(f"Unknown normalization method: {method}")
        
        self.df = df_normalized
        return self

    def get_cleaned_data(self):
        return self.df

    def get_cleaning_report(self):
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': removed_rows,
            'columns_removed': removed_cols,
            'missing_values_remaining': self.df.isnull().sum().sum()
        }
        
        return report

def load_and_clean_data(filepath, cleaning_steps=None):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            if step['method'] == 'handle_missing':
                cleaner.handle_missing_values(**step.get('params', {}))
            elif step['method'] == 'remove_outliers':
                cleaner.remove_outliers_iqr(**step.get('params', {}))
            elif step['method'] == 'normalize':
                cleaner.normalize_data(**step.get('params', {}))
    
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import sys

def remove_duplicates(input_file, output_file, subset=None):
    """
    Reads a CSV file, removes duplicate rows, and saves the cleaned data.
    If 'subset' is provided, only considers those columns for duplicate detection.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(subset=subset, keep='first')
        final_count = len(df_cleaned)
        df_cleaned.to_csv(output_file, index=False)
        print(f"Successfully removed {initial_count - final_count} duplicate rows.")
        print(f"Cleaned data saved to '{output_file}'.")
        return df_cleaned
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv> [subset_columns]")
        print("Example: python data_cleaner.py raw_data.csv cleaned_data.csv")
        print("Example with subset: python data_cleaner.py raw_data.csv cleaned_data.csv 'column1,column2'")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    subset_cols = None
    if len(sys.argv) == 4:
        subset_cols = [col.strip() for col in sys.argv[3].split(',')]

    remove_duplicates(input_csv, output_csv, subset_cols)
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result