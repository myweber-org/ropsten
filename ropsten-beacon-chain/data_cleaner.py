
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict): Optional dictionary to rename columns
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Drop duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    if fill_missing:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                # Fill numeric columns with median
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif cleaned_df[col].dtype == 'object':
                # Fill categorical columns with mode
                mode_value = cleaned_df[col].mode()
                if not mode_value.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value.iloc[0])
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the cleaned DataFrame for basic data quality.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
        validation_results['all_required_columns_present'] = len(missing_cols) == 0
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with some issues
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing=True)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    validation = validate_data(cleaned, required_columns=['id', 'name', 'age', 'score'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        if self.file_path.exists():
            self.df = pd.read_csv(self.file_path)
        else:
            raise FileNotFoundError(f"File not found: {self.file_path}")
    
    def remove_duplicates(self):
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_count - len(self.df)
        return removed
    
    def fill_missing_values(self, strategy='mean', custom_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'custom' and custom_value is not None:
                    fill_value = custom_value
                else:
                    continue
                
                self.df[col].fillna(fill_value, inplace=True)
        
        return len(numeric_cols)
    
    def remove_outliers(self, method='iqr', threshold=1.5):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                initial_count = len(self.df)
                self.df = self.df[(self.df[col] >= lower_bound) & 
                                 (self.df[col] <= upper_bound)]
                outliers_removed += (initial_count - len(self.df))
        
        return outliers_removed
    
    def standardize_columns(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].std() != 0:
                self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
        
        return len(numeric_cols)
    
    def save_cleaned_data(self, output_path=None):
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        return output_path
    
    def get_summary(self):
        summary = {
            'original_file': str(self.file_path),
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def process_csv_file(input_file, output_dir='cleaned_data'):
    cleaner = DataCleaner(input_file)
    
    summary = cleaner.get_summary()
    duplicates_removed = cleaner.remove_duplicates()
    columns_filled = cleaner.fill_missing_values(strategy='median')
    outliers_removed = cleaner.remove_outliers()
    columns_standardized = cleaner.standardize_columns()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = cleaner.save_cleaned_data(output_dir / f"cleaned_{Path(input_file).name}")
    
    results = {
        'summary': summary,
        'duplicates_removed': duplicates_removed,
        'columns_with_filled_missing': columns_filled,
        'outliers_removed': outliers_removed,
        'columns_standardized': columns_standardized,
        'output_file': str(output_path)
    }
    
    return results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [100, 200, None, 400, 500, 600, 700, 800, 900, 1000],
        'D': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    
    results = process_csv_file(test_file)
    
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    Path(test_file).unlink(missing_ok=True)
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
    
    Returns:
        pd.Series: Normalized column values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_min = df[column].min()
    col_max = df[column].max()
    
    if col_max == col_min:
        return pd.Series([0] * len(df), index=df.index)
    
    normalized = (df[column] - col_min) / (col_max - col_min)
    return normalized