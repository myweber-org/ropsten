import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def fill_missing(self, strategy: str = 'mean', custom_values: Optional[Dict] = None) -> 'DataCleaner':
        if custom_values:
            self.df = self.df.fillna(custom_values)
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else None)
        return self
        
    def remove_outliers(self, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
        
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except (ValueError, TypeError):
                    continue
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': int(self.df.isnull().sum().sum()),
            'duplicates_removed': self.original_shape[0] - self.df.drop_duplicates().shape[0]
        }

def clean_csv_file(input_path: str, output_path: str, cleaning_steps: Optional[Dict] = None) -> Dict:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        if cleaning_steps.get('remove_duplicates'):
            cleaner.remove_duplicates(cleaning_steps.get('duplicate_subset'))
        if cleaning_steps.get('fill_missing'):
            cleaner.fill_missing(
                strategy=cleaning_steps.get('fill_strategy', 'mean'),
                custom_values=cleaning_steps.get('custom_fill_values')
            )
        if cleaning_steps.get('remove_outliers'):
            cleaner.remove_outliers(
                columns=cleaning_steps.get('outlier_columns', []),
                method=cleaning_steps.get('outlier_method', 'iqr'),
                threshold=cleaning_steps.get('outlier_threshold', 1.5)
            )
        if cleaning_steps.get('convert_types'):
            cleaner.convert_types(cleaning_steps.get('type_mapping', {}))
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_cleaning_report()import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_na (bool): If True, drop rows with any null values. If False, fill with column mean.
    column_case (str): Desired case for column names ('lower', 'upper', or 'title')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if drop_na:
        df_clean = df_clean.dropna()
    else:
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Standardize column names
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    # Remove any leading/trailing whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'Name': ['Alice', 'Bob', None, 'Charlie'],
#         'Age': [25, None, 30, 35],
#         'Score': [85.5, 92.0, 78.5, None]
#     })
#     
#     cleaned = clean_dataset(sample_data, drop_na=False, column_case='lower')
#     print("Original DataFrame:")
#     print(sample_data)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataframe(cleaned, required_columns=['name', 'age'])
#     print(f"\nValidation: {message}")