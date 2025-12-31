
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                clean_df = clean_df[z_scores < threshold]
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        return normalized_df
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        standardized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = standardized_df[col].mean()
                std_val = standardized_df[col].std()
                if std_val > 0:
                    standardized_df[col] = (standardized_df[col] - mean_val) / std_val
        return standardized_df
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    data['feature_a'][[10, 25, 50]] = [500, -200, 300]
    data['feature_b'][[15, 30]] = [1000, 1200]
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    cleaner = DataCleaner(df)
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    print(f"Original shape: {summary['original_shape']}")
    print(f"Numeric columns: {summary['numeric_columns']}")
    
    clean_iqr = cleaner.remove_outliers_iqr()
    print(f"\nAfter IQR outlier removal: {clean_iqr.shape}")
    
    clean_zscore = cleaner.remove_outliers_zscore()
    print(f"After Z-score outlier removal: {clean_zscore.shape}")
    
    normalized = cleaner.normalize_minmax()
    print(f"\nMin-Max normalized data range:")
    for col in cleaner.numeric_columns:
        print(f"{col}: [{normalized[col].min():.3f}, {normalized[col].max():.3f}]")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main cleaning function for datasets
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, col)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, check_nulls=True):
    """
    Validate dataframe structure and content
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if check_nulls:
        null_counts = df.isnull().sum()
        if null_counts.any():
            print("Warning: DataFrame contains null values")
            print(null_counts[null_counts > 0])
    
    return True

def get_data_summary(df):
    """
    Generate comprehensive data summary
    """
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
        'categorical_stats': {col: df[col].value_counts().to_dict() 
                            for col in df.select_dtypes(include=['object']).columns}
    }
    return summary

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned data to file
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {output_path}")import numpy as np
import pandas as pd

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to check for outliers
        threshold: multiplier for IQR (default 1.5)
    
    Returns:
        Boolean mask indicating outliers
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns=None, threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        data: pandas DataFrame
        columns: list of column names or None for all numeric columns
        threshold: IQR threshold multiplier
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    outlier_mask = pd.Series(False, index=data.index)
    
    for col in columns:
        if col in data.columns:
            outlier_mask |= detect_outliers_iqr(data, col, threshold)
    
    return data[~outlier_mask].copy()

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling to [0, 1] range.
    
    Args:
        data: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        Normalized DataFrame
    """
    normalized_data = data.copy()
    
    if columns is None:
        columns = normalized_data.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in normalized_data.columns:
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            
            if col_max != col_min:
                normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def standardize_zscore(data, columns=None):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        Standardized DataFrame
    """
    standardized_data = data.copy()
    
    if columns is None:
        columns = standardized_data.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in standardized_data.columns:
            col_mean = standardized_data[col].mean()
            col_std = standardized_data[col].std()
            
            if col_std != 0:
                standardized_data[col] = (standardized_data[col] - col_mean) / col_std
            else:
                standardized_data[col] = 0
    
    return standardized_data

def clean_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in the dataset.
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of column names or None for all columns
    
    Returns:
        DataFrame with missing values handled
    """
    cleaned_data = data.copy()
    
    if columns is None:
        columns = cleaned_data.columns
    
    for col in columns:
        if col not in cleaned_data.columns:
            continue
            
        if cleaned_data[col].isnull().any():
            if strategy == 'drop':
                cleaned_data = cleaned_data.dropna(subset=[col])
            elif strategy == 'mean':
                cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
            elif strategy == 'median':
                cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
            elif strategy == 'mode':
                cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate a summary of the dataset including missing values and basic statistics.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        Dictionary containing data summary
    """
    summary = {
        'shape': data.shape,
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            '25%': data[col].quantile(0.25),
            '50%': data[col].median(),
            '75%': data[col].quantile(0.75),
            'max': data[col].max()
        }
    
    return summary
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = df[col].mean()
                elif fill_missing == 'median':
                    fill_value = df[col].median()
                elif fill_missing == 'zero':
                    fill_value = 0
                else:
                    fill_value = fill_missing
                
                df[col] = df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_value}")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
            print(f"Filled missing values in column '{col}' with 'Unknown'")
    
    print(f"Dataset cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset for required columns and data integrity.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            print(f"Warning: Column '{col}' still contains missing values")
    
    return True

def main():
    """
    Example usage of the data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.0, 20.0, np.nan, 30.0],
        'category': ['A', 'B', 'B', None, 'C', 'A'],
        'score': [85, 90, 90, 95, None, 100]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    try:
        validate_dataset(cleaned_df, required_columns=['id', 'value', 'category'])
        print("\nDataset validation passed")
    except ValueError as e:
        print(f"\nDataset validation failed: {e}")

if __name__ == "__main__":
    main()