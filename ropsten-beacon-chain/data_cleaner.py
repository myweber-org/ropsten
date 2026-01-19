
import pandas as pd
import re

def clean_string_column(series, case='lower', strip=True, remove_special=True):
    """
    Standardize string values in a pandas Series.
    
    Args:
        series (pd.Series): Input series containing string data.
        case (str): Desired case transformation. Options: 'lower', 'upper', 'title', None.
        strip (bool): Whether to strip leading/trailing whitespace.
        remove_special (bool): Whether to remove special characters (keeping alphanumeric and spaces).
    
    Returns:
        pd.Series: Cleaned series.
    """
    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)
    
    result = series.copy()
    
    if strip:
        result = result.str.strip()
    
    if remove_special:
        result = result.apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x) if pd.notnull(x) else x)
    
    if case == 'lower':
        result = result.str.lower()
    elif case == 'upper':
        result = result.str.upper()
    elif case == 'title':
        result = result.str.title()
    
    return result

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame with additional logging.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Columns to consider for duplicates.
        keep (str): Which duplicates to keep. Options: 'first', 'last', False.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    final_count = len(df_clean)
    
    duplicates_removed = initial_count - final_count
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows.")
    
    return df_clean

def validate_email_format(series):
    """
    Validate email format in a pandas Series.
    
    Args:
        series (pd.Series): Series containing email addresses.
    
    Returns:
        pd.Series: Boolean series indicating valid emails.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return series.str.match(pattern, na=False)

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'name': ['  John Doe  ', 'Jane Smith', 'ALICE WONDER', 'bob@example'],
        'email': ['john@example.com', 'invalid-email', 'alice@company.co.uk', 'bob@test.org'],
        'value': [100, 200, 100, 300]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df['name_clean'] = clean_string_column(df['name'], case='title', strip=True, remove_special=True)
    df['email_valid'] = validate_email_format(df['email'])
    
    print("After cleaning:")
    print(df)
    print()
    
    df_no_dupes = remove_duplicates(df, subset=['value'], keep='first')
    print("After removing duplicates by 'value' column:")
    print(df_no_dupes)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataframe(df: pd.DataFrame, 
                    drop_duplicates: bool = True,
                    columns_to_standardize: Optional[List[str]] = None,
                    date_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, standardizing text columns,
    and converting date columns to datetime format.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    if columns_to_standardize:
        for col in columns_to_standardize:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
                df_clean[col] = df_clean[col].replace({'nan': np.nan, 'none': np.nan})
    
    if date_columns:
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    return df_clean

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def calculate_missing_percentage(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the percentage of missing values for each column.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / total_rows) * 100
    
    return missing_percentage.round(2)import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(fill_value)
        else:
            raise ValueError("Invalid strategy or missing fill_value for constant strategy")
        self.df[self.categorical_columns] = self.df[self.categorical_columns].fillna('Unknown')
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self

    def get_cleaned_data(self):
        return self.df

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', np.nan, 'z', 'x']
    }
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .handle_missing_values(strategy='mean')
                  .remove_outliers_iqr(multiplier=1.5)
                  .standardize_data()
                  .get_cleaned_data())
    print(cleaned_df)

if __name__ == "__main__":
    example_usage()import pandas as pd
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
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
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of numeric column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal summary statistics:")
    for col in ['A', 'B', 'C']:
        stats = calculate_summary_statistics(df, col)
        print(f"\n{col}: {stats}")
    
    cleaned_df = clean_dataset(df, ['A', 'B', 'C'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    for col in ['A', 'B', 'C']:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"\n{col}: {stats}")