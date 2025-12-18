import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            if fill_value is not None:
                self.df.fillna(fill_value, inplace=True)
            else:
                raise ValueError("fill_value must be provided for constant strategy")
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'constant'")
        
        return self.df

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        cleaned_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
                cleaned_df = cleaned_df[mask]
        
        self.df = cleaned_df.reset_index(drop=True)
        return self.df

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        standardized_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean = standardized_df[col].mean()
                std = standardized_df[col].std()
                if std > 0:
                    standardized_df[col] = (standardized_df[col] - mean) / std
        
        self.df = standardized_df
        return self.df

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        self.df = normalized_df
        return self.df

    def get_cleaned_data(self):
        return self.df.copy()

def create_sample_data():
    data = {
        'age': [25, 30, np.nan, 35, 40, 200, 45, 50, 55, 60],
        'salary': [50000, 60000, 70000, np.nan, 90000, 1000000, 110000, 120000, 130000, 140000],
        'department': ['HR', 'IT', 'IT', 'HR', 'Finance', 'Finance', 'IT', 'HR', 'IT', 'Finance']
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    print("Original Data:")
    print(df)
    print("\n" + "="*50)
    
    cleaner = DataCleaner(df)
    
    cleaned_df = cleaner.handle_missing_values(strategy='mean')
    print("After handling missing values:")
    print(cleaned_df)
    print("\n" + "="*50)
    
    cleaned_df = cleaner.remove_outliers_iqr(multiplier=1.5)
    print("After removing outliers:")
    print(cleaned_df)
    print("\n" + "="*50)
    
    cleaned_df = cleaner.standardize_data()
    print("After standardization:")
    print(cleaned_df)
    print("\n" + "="*50)
    
    final_data = cleaner.get_cleaned_data()
    print("Final cleaned data:")
    print(final_data)