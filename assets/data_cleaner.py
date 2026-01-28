
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str or dict): Method to fill missing values.
        If a string, it can be 'mean', 'median', 'mode', or a constant value.
        If a dict, specify column-wise fill values.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)

    return cleaned_df

def normalize_column(df, column, method='minmax'):
    """
    Normalize a specific column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to normalize.
    method (str): Normalization method ('minmax' or 'zscore').

    Returns:
    pd.DataFrame: DataFrame with the normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    normalized_df = df.copy()

    if method == 'minmax':
        min_val = normalized_df[column].min()
        max_val = normalized_df[column].max()
        if max_val != min_val:
            normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
        else:
            normalized_df[column] = 0
    elif method == 'zscore':
        mean_val = normalized_df[column].mean()
        std_val = normalized_df[column].std()
        if std_val != 0:
            normalized_df[column] = (normalized_df[column] - mean_val) / std_val
        else:
            normalized_df[column] = 0
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'.")

    return normalized_df
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
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
                    fill_value = strategy
                    
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
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
            
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
    
    def get_cleaning_report(self):
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': removed_rows,
            'columns_removed': removed_cols,
            'missing_values': self.df.isnull().sum().sum()
        }
        
        return report
    
    def save_cleaned_data(self, filepath):
        self.df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_outliers_iqr(multiplier=1.5)
    
    report = cleaner.get_cleaning_report()
    print(f"Cleaning report: {report}")
    
    return cleaner.df

if __name__ == "__main__":
    cleaned_df = example_usage()
    print(f"Cleaned DataFrame:\n{cleaned_df}")