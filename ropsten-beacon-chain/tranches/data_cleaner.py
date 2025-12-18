
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    date_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and converting data types.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
        date_columns: List of column names to parse as dates
    
    Returns:
        Cleaned DataFrame
    """
    
    # Read CSV file
    df = pd.read_csv(input_path)
    
    # Parse date columns if specified
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Handle missing values based on strategy
    if missing_strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == 'zero':
        df = df.fillna(0)
    elif missing_strategy == 'drop':
        df = df.dropna()
    
    # Convert object columns to categorical if they have few unique values
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 10:
            df[col] = df[col].astype('category')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['category']).columns),
        'date_columns': list(df.select_dtypes(include=['datetime']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    cleaned_df = clean_csv_data(
        input_path='raw_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean',
        date_columns=['transaction_date', 'created_at']
    )
    
    validation = validate_dataframe(cleaned_df)
    print(f"Data cleaning completed. Validation results: {validation}")import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                mean_val = self.df[col].mean()
                df_filled[col] = self.df[col].fillna(mean_val)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'current_columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature1'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature2'] = 1000
    
    cleaner = DataCleaner(df)
    print(f"Initial shape: {cleaner.original_shape}")
    
    removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {removed} outliers")
    
    cleaner.fill_missing_mean()
    cleaner.standardize_zscore(['feature1', 'feature2'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Final shape: {cleaned_df.shape}")
    print(f"Summary: {summary}")
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.data = None
        
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix == '.csv':
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format")
            
        return self.data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.data is None:
            self.load_data()
        
        if columns is None:
            columns = self.data.columns
        
        for column in columns:
            if column in self.data.columns:
                if self.data[column].isnull().any():
                    if strategy == 'mean':
                        fill_value = self.data[column].mean()
                    elif strategy == 'median':
                        fill_value = self.data[column].median()
                    elif strategy == 'mode':
                        fill_value = self.data[column].mode()[0]
                    elif strategy == 'drop':
                        self.data = self.data.dropna(subset=[column])
                        continue
                    else:
                        fill_value = strategy
                    
                    self.data[column] = self.data[column].fillna(fill_value)
        
        return self.data
    
    def remove_duplicates(self, subset=None):
        if self.data is None:
            self.load_data()
        
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset)
        removed = initial_rows - len(self.data)
        
        return self.data, removed
    
    def normalize_column(self, column, method='minmax'):
        if self.data is None:
            self.load_data()
        
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if method == 'minmax':
            col_min = self.data[column].min()
            col_max = self.data[column].max()
            if col_max != col_min:
                self.data[column] = (self.data[column] - col_min) / (col_max - col_min)
        
        elif method == 'zscore':
            col_mean = self.data[column].mean()
            col_std = self.data[column].std()
            if col_std != 0:
                self.data[column] = (self.data[column] - col_mean) / col_std
        
        return self.data
    
    def save_cleaned_data(self, output_path=None):
        if self.data is None:
            raise ValueError("No data to save. Load and clean data first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.csv':
            self.data.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.data.to_excel(output_path, index=False)
        
        return output_path

def clean_dataset(input_file, output_file=None, missing_strategy='mean'):
    cleaner = DataCleaner(input_file)
    cleaner.load_data()
    cleaner.handle_missing_values(strategy=missing_strategy)
    cleaner.remove_duplicates()
    cleaner.save_cleaned_data(output_file)
    
    return cleaner.data
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    column (str): Column name
    
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
        'count': df[column].count()
    }
    
    return stats

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100)
    }
    
    # Add some outliers
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[100] = [101, 500]  # High outlier
    sample_df.loc[101] = [102, -50]  # Low outlier
    
    print("Original DataFrame shape:", sample_df.shape)
    print("Original summary stats:", calculate_summary_stats(sample_df, 'value'))
    
    # Clean data
    cleaned_df = remove_outliers_iqr(sample_df, 'value')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary stats:", calculate_summary_stats(cleaned_df, 'value'))
    
    # Validate
    is_valid, message = validate_dataframe(cleaned_df, ['id', 'value'])
    print(f"\nValidation: {is_valid} - {message}")