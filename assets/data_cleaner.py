
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
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Dataset contains {nan_count} NaN values")
    
    return True
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in df_normalized.columns:
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(df_normalized[col])
                elif method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = df_normalized[col].median()
                    iqr = df_normalized[col].quantile(0.75) - df_normalized[col].quantile(0.25)
                    if iqr != 0:
                        df_normalized[col] = (df_normalized[col] - median) / iqr
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                elif strategy == 'constant':
                    fill_value = 0
                else:
                    continue
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'feature1'] = np.nan
    df.loc[5, 'feature2'] = 1000
    
    cleaner = DataCleaner(df)
    print("Initial summary:", cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {removed} outliers")
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.normalize_data(method='zscore')
    
    print("Final summary:", cleaner.get_summary())
    cleaned_df = cleaner.get_cleaned_data()
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("Cleaning completed successfully")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def main():
    # Example usage
    import pandas as pd
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'value': np.random.randn(100)
    })
    sample_data.loc[0, 'value'] = 10  # Add an outlier
    
    print("Original data shape:", sample_data.shape)
    cleaned_data = remove_outliers_iqr(sample_data, 'value')
    print("Cleaned data shape:", cleaned_data.shape)
    print("Outliers removed:", sample_data.shape[0] - cleaned_data.shape[0])

if __name__ == "__main__":
    main()