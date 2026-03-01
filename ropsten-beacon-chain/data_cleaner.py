
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mean':
                    fill_value = self.df[col].mean()
                elif method == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self
    
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_mask = pd.Series([False] * len(self.df))
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            col_outliers = (z_scores > threshold)
            outliers_mask = outliers_mask | col_outliers.reindex(self.df.index, fill_value=False)
        
        return outliers_mask
    
    def remove_outliers(self, threshold=3):
        outliers_mask = self.detect_outliers_zscore(threshold)
        self.df = self.df[~outliers_mask]
        return self
    
    def normalize_numeric(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_report(self):
        removed_rows = self.original_shape[0] - len(self.df)
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': removed_rows,
            'columns_removed': removed_cols,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(exclude=[np.number]).columns)
        }
        
        return report

def clean_dataset(df, remove_outliers=True, normalize=True):
    cleaner = DataCleaner(df)
    
    cleaner.remove_missing(threshold=0.3)
    cleaner.fill_numeric_missing(method='median')
    
    if remove_outliers:
        cleaner.remove_outliers(threshold=3)
    
    if normalize:
        cleaner.normalize_numeric(method='standard')
    
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', remove_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    file_path (str): Path to the CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    remove_threshold (float): Remove columns with missing ratio above this threshold
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Cleaning statistics
    """
    
    df = pd.read_csv(file_path)
    original_shape = df.shape
    
    stats = {
        'original_rows': original_shape[0],
        'original_columns': original_shape[1],
        'missing_values': df.isnull().sum().sum(),
        'cleaning_applied': []
    }
    
    for column in df.columns:
        missing_ratio = df[column].isnull().sum() / len(df)
        
        if missing_ratio > remove_threshold:
            df = df.drop(columns=[column])
            stats['cleaning_applied'].append(f'Removed column {column} (missing: {missing_ratio:.1%})')
            continue
            
        if df[column].isnull().any():
            if fill_method == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].mean()
            elif fill_method == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].median()
            elif fill_method == 'mode':
                fill_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
            elif fill_method == 'zero':
                fill_value = 0
            else:
                fill_value = df[column].ffill().bfill().iloc[0] if not df[column].isnull().all() else np.nan
            
            df[column] = df[column].fillna(fill_value)
            stats['cleaning_applied'].append(f'Filled missing values in {column} using {fill_method}')
    
    stats['final_rows'] = df.shape[0]
    stats['final_columns'] = df.shape[1]
    stats['remaining_missing'] = df.isnull().sum().sum()
    
    return df, stats

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    numeric_columns (list): List of columns that should be numeric
    
    Returns:
    bool: True if validation passes
    list: List of validation errors
    """
    
    errors = []
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f'Missing required columns: {missing_columns}')
    
    if numeric_columns:
        for column in numeric_columns:
            if column in df.columns and not pd.api.types.is_numeric_dtype(df[column]):
                errors.append(f'Column {column} should be numeric but has type {df[column].dtype}')
    
    if df.empty:
        errors.append('DataFrame is empty')
    
    return len(errors) == 0, errors

def export_cleaned_data(df, output_path, format='csv'):
    """
    Export cleaned DataFrame to file.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame
    output_path (str): Path for output file
    format (str): Output format ('csv', 'excel', 'json')
    """
    
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 10, 11],
        'C': ['x', 'y', 'z', np.nan, 'w'],
        'D': [100, 200, 300, 400, 500]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df, cleaning_stats = clean_csv_data('sample_data.csv', fill_method='mean')
    
    print(f"Original shape: {cleaning_stats['original_rows']}x{cleaning_stats['original_columns']}")
    print(f"Final shape: {cleaning_stats['final_rows']}x{cleaning_stats['final_columns']}")
    print(f"Missing values removed: {cleaning_stats['missing_values']}")
    
    is_valid, validation_errors = validate_dataframe(
        cleaned_df, 
        required_columns=['A', 'D'],
        numeric_columns=['A', 'D']
    )
    
    if is_valid:
        export_cleaned_data(cleaned_df, 'cleaned_data.csv')
        print("Data cleaning completed successfully")
    else:
        print(f"Validation errors: {validation_errors}")