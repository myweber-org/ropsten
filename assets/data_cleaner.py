
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
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
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
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

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_missing(self, threshold=0.8):
        for col in self.df.columns:
            missing_ratio = self.df[col].isna().sum() / len(self.df)
            if missing_ratio > threshold:
                self.df.drop(columns=[col], inplace=True)
            else:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        return self
    
    def remove_outliers(self, z_threshold=3):
        for col in self.numeric_columns:
            if col in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df[col]))
                self.df = self.df[z_scores < z_threshold]
        return self
    
    def normalize_data(self, method='minmax'):
        if method == 'minmax':
            for col in self.numeric_columns:
                if col in self.df.columns:
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val > min_val:
                        self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            for col in self.numeric_columns:
                if col in self.df.columns:
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val > 0:
                        self.df[col] = (self.df[col] - mean_val) / std_val
        return self
    
    def get_cleaned_data(self):
        return self.df.reset_index(drop=True)

def process_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        cleaner = DataCleaner(df)
        cleaned_df = (cleaner.remove_missing(threshold=0.7)
                              .remove_outliers(z_threshold=2.5)
                              .normalize_data(method='standard')
                              .get_cleaned_data())
        return cleaned_df
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame containing the data
        column: string name of the column to clean
    
    Returns:
        DataFrame with outliers removed from specified column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Args:
        data: pandas DataFrame
        column: string name of the column
    
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Args:
        data: pandas DataFrame
        columns_to_clean: list of column names to clean
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_dataimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (str or dict): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
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
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"
import numpy as np
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers using IQR method")
        return self
        
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(self.df[col])
                elif method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = self.df[col].median()
                    iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
                    if iqr != 0:
                        df_normalized[col] = (self.df[col] - median) / iqr
        
        self.df = df_normalized
        print(f"Applied {method} normalization to selected columns")
        return self
        
    def handle_missing_values(self, strategy='mean', fill_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        
        self.df[categorical_cols] = self.df[categorical_cols].fillna('Unknown')
        print(f"Handled missing values using {strategy} strategy")
        return self
        
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(1000, 50, replace=False)
    for idx in indices:
        col = np.random.choice(['feature_a', 'feature_b', 'feature_c'])
        df.loc[idx, col] = np.nan
    
    outlier_indices = np.random.choice(1000, 20, replace=False)
    for idx in outlier_indices:
        col = np.random.choice(['feature_a', 'feature_b'])
        df.loc[idx, col] *= 5
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    cleaner.handle_missing_values(strategy='median')
    cleaner.remove_outliers_iqr(threshold=1.5)
    cleaner.normalize_data(method='zscore')
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())