
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else '', inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean())
        print(f"Missing values handled using '{strategy}' strategy")
        return self.df
    
    def normalize_column(self, column: str) -> pd.DataFrame:
        if column in self.df.columns:
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        return self.df
    
    def get_summary(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'current_rows': len(self.df),
            'original_columns': self.original_shape[1],
            'current_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates_removed': self.original_shape[0] - len(self.df)
        }

def clean_dataset(filepath: str, output_path: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy='fill')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaner.normalize_column(col)
    
    summary = cleaner.get_summary()
    print("Cleaning summary:", summary)
    
    if output_path:
        cleaner.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    
    return cleaner.df
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Fill missing numeric values with column median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if not df_cleaned[col].mode().empty:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna('Unknown')

    return df_cleaned

def validate_data(df):
    """
    Validate that the cleaned DataFrame has no missing values.
    """
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("Data validation passed: No missing values found.")
        return True
    else:
        print(f"Data validation failed: {missing_values} missing values found.")
        return False

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)
    
    # Validate the cleaned data
    validate_data(cleaned_df)
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (default: all columns)
        fill_method: method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    duplicates_removed = initial_rows - len(cleaned_df)
    
    # Handle missing values
    missing_before = cleaned_df.isnull().sum().sum()
    
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_method == 'mean':
                fill_value = cleaned_df[col].mean()
            else:  # median
                fill_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(fill_value)
    elif fill_method == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_value = cleaned_df[col].mode()
                if not mode_value.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value.iloc[0])
    
    missing_after = cleaned_df.isnull().sum().sum()
    
    # Print cleaning summary
    print(f"Data cleaning completed:")
    print(f"- Removed {duplicates_removed} duplicate rows")
    print(f"- Filled {missing_before - missing_after} missing values")
    print(f"- Remaining missing values: {missing_after}")
    print(f"- Original shape: {df.shape}")
    print(f"- Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of columns that must be present
        min_rows: minimum number of rows required
    
    Returns:
        Boolean indicating if validation passed
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with some issues
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'Eve', 'Eve'],
        'age': [25, 30, 30, None, 35, 28, 28],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0, 88.0],
        'department': ['Sales', 'IT', 'IT', 'HR', 'Sales', 'IT', 'IT']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, fill_method='mean')
    
    print("\n" + "="*50 + "\n")
    print("Cleaned DataFrame:")
    print(cleaned)