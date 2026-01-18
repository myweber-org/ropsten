
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def calculate_summary_statistics(df):
    summary = df.describe()
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 100, 200)
    })
    print("Original dataset shape:", sample_data.shape)
    cleaned_data = clean_dataset(sample_data, ['A', 'B', 'C'])
    print("Cleaned dataset shape:", cleaned_data.shape)
    stats = calculate_summary_statistics(cleaned_data)
    print("\nSummary statistics:")
    print(stats)
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_clean (list, optional): List of column names to apply string normalization.
            If None, all object dtype columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # Normalize string columns
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(_normalize_string)
    
    print(f"Removed {removed_duplicates} duplicate rows.")
    print(f"Normalized columns: {columns_to_clean}")
    
    return df_clean

def _normalize_string(value):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters from the edges.
    
    Args:
        value: Input value (will be converted to string if not already).
    
    Returns:
        str: Normalized string, or original value if not a string.
    """
    if not isinstance(value, str):
        return value
    
    # Convert to lowercase
    normalized = value.lower()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

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
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    df_validated = df.copy()
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_validated['email_valid'] = df_validated[email_column].apply(
        lambda x: bool(re.match(email_regex, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = df_validated['email_valid'].sum()
    total_count = len(df_validated)
    
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return df_validated

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'name': ['John Doe', 'Jane Smith', 'john doe', 'Alice Johnson', 'Jane Smith'],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'alice@example.org', 'jane@example.com'],
        'age': [25, 30, 25, 35, 30]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    df_clean = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(df_clean)
    print("\n")
    
    # Validate emails
    df_validated = validate_email_column(df_clean, 'email')
    print("\nDataFrame with email validation:")
    print(df_validated)
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} not found")
        
        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format")
        return self
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
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
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    fill_value = strategy
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self
    
    def normalize_column(self, column, method='minmax'):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in dataframe")
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val != 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        
        return self
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data to save")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        else:
            raise ValueError("Unsupported output format")
        
        return output_path
    
    def get_summary(self):
        if self.df is None:
            return {}
        
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum(),
            'column_types': self.df.dtypes.to_dict()
        }
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    cleaner.load_data()
    cleaner.handle_missing_values()
    cleaner.remove_duplicates()
    
    if output_file:
        return cleaner.save_cleaned_data(output_file)
    else:
        return cleaner.save_cleaned_data()