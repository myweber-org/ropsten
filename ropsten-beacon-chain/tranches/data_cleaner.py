
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using Z-score method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Z-score normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    z_scores = (data[column] - mean_val) / std_val
    return z_scores

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data column.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to check
    threshold (float): Absolute skewness threshold
    
    Returns:
    tuple: (skewness_value, is_skewed)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    skewness = stats.skew(data[column].dropna())
    is_skewed = abs(skewness) > threshold
    
    return skewness, is_skewed

def create_clean_dataframe(data, numeric_columns, outlier_multiplier=1.5):
    """
    Create cleaned dataframe with outlier removal and normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_multiplier (float): IQR multiplier for outlier removal
    
    Returns:
    pd.DataFrame: Cleaned dataframe with normalized values
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            original_len = len(cleaned_data)
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            removed_count = original_len - len(cleaned_data)
            
            if removed_count > 0:
                normalized_col = normalize_minmax(cleaned_data, column)
                cleaned_data[f"{column}_normalized"] = normalized_col
    
    return cleaned_data

def summarize_cleaning(data, cleaned_data, numeric_columns):
    """
    Generate summary statistics for cleaning process.
    
    Parameters:
    data (pd.DataFrame): Original dataframe
    cleaned_data (pd.DataFrame): Cleaned dataframe
    numeric_columns (list): List of processed numeric columns
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'original_rows': len(data),
        'cleaned_rows': len(cleaned_data),
        'removed_rows': len(data) - len(cleaned_data),
        'removed_percentage': ((len(data) - len(cleaned_data)) / len(data)) * 100,
        'columns_processed': []
    }
    
    for column in numeric_columns:
        if column in data.columns and column in cleaned_data.columns:
            col_summary = {
                'column': column,
                'original_mean': data[column].mean(),
                'cleaned_mean': cleaned_data[column].mean(),
                'original_std': data[column].std(),
                'cleaned_std': cleaned_data[column].std(),
                'skewness_original': stats.skew(data[column].dropna()),
                'skewness_cleaned': stats.skew(cleaned_data[column].dropna())
            }
            summary['columns_processed'].append(col_summary)
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    cleaned = create_clean_dataframe(sample_data, numeric_cols)
    summary_stats = summarize_cleaning(sample_data, cleaned, numeric_cols)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned.shape}")
    print(f"Rows removed: {summary_stats['removed_rows']}")
    print(f"Removed percentage: {summary_stats['removed_percentage']:.2f}%")
    
    for col_summary in summary_stats['columns_processed']:
        print(f"\nColumn: {col_summary['column']}")
        print(f"  Original mean: {col_summary['original_mean']:.2f}")
        print(f"  Cleaned mean: {col_summary['cleaned_mean']:.2f}")
        print(f"  Skewness change: {col_summary['skewness_original']:.2f} -> {col_summary['skewness_cleaned']:.2f}")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    cleaned_df = dataframe.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data_minmax(dataframe, columns):
    normalized_df = dataframe.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def clean_dataset(file_path, output_path=None):
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("No numeric columns found for processing")
            return df
        
        print(f"Original shape: {df.shape}")
        df_clean = remove_outliers_iqr(df, numeric_cols)
        print(f"After outlier removal: {df_clean.shape}")
        
        df_normalized = normalize_data_minmax(df_clean, numeric_cols)
        
        if output_path:
            df_normalized.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df_normalized
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    cleaned_data = clean_dataset("input_data.csv", "cleaned_data.csv")import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing columns
    with excessive missing data.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values.
                         Options: 'mean', 'median', 'mode', 'zero'.
    drop_threshold (float): Threshold for column removal (0 to 1).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    df = pd.read_csv(filepath)
    
    missing_percentage = df.isnull().sum() / len(df)
    columns_to_drop = missing_percentage[missing_percentage > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        fill_values = df[numeric_columns].mean()
    elif fill_strategy == 'median':
        fill_values = df[numeric_columns].median()
    elif fill_strategy == 'mode':
        fill_values = df[numeric_columns].mode().iloc[0]
    elif fill_strategy == 'zero':
        fill_values = 0
    else:
        raise ValueError("Invalid fill_strategy. Choose from 'mean', 'median', 'mode', 'zero'.")
    
    df[numeric_columns] = df[numeric_columns].fillna(fill_values)
    
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    df[non_numeric_columns] = df[non_numeric_columns].fillna('Unknown')
    
    return df

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame.
    output_path (str): Path for output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data exported to {output_path}")

if __name__ == "__main__":
    cleaned_df = clean_csv_data('raw_data.csv', fill_strategy='median', drop_threshold=0.3)
    export_cleaned_data(cleaned_df, 'cleaned_data.csv')
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
        
        self.data = clean_data.reset_index(drop=True)
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.data.columns:
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                if max_val != min_val:
                    self.data[col] = (self.data[col] - min_val) / (max_val - min_val)
        
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.data.columns:
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                if std_val > 0:
                    self.data[col] = (self.data[col] - mean_val) / std_val
        
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.data.columns and self.data[col].isnull().any():
                median_val = self.data[col].median()
                self.data[col] = self.data[col].fillna(median_val)
        
        return self
    
    def get_cleaned_data(self):
        return self.data
    
    def get_removed_count(self):
        return self.original_shape[0] - self.data.shape[0]

def create_sample_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    data.loc[np.random.choice(1000, 50), 'feature_a'] = np.nan
    data.loc[np.random.choice(1000, 20), 'feature_b'] = 1000
    
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    print(f"Original data shape: {sample_data.shape}")
    
    cleaned = (cleaner
               .fill_missing_median()
               .remove_outliers_iqr(['feature_a', 'feature_b'])
               .normalize_minmax(['feature_a', 'feature_c'])
               .standardize_zscore(['feature_b'])
               .get_cleaned_data())
    
    print(f"Cleaned data shape: {cleaned.shape}")
    print(f"Rows removed: {cleaner.get_removed_count()}")
    print(f"Missing values after cleaning: {cleaned.isnull().sum().sum()}")
    
    print("\nSummary statistics:")
    print(cleaned.describe())
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and save the cleaned data.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove leading/trailing whitespace from string columns
        for col in categorical_cols:
            df[col] = df[col].str.strip()
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Save cleaned data
        if output_path is None:
            output_path = Path(input_path).stem + '_cleaned.csv'
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Original rows: {len(pd.read_csv(input_path))}, Cleaned rows: {len(df)}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Perform basic validation on the dataframe.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    print("Data Validation Report:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nNumeric columns statistics:")
        print(df[numeric_cols].describe())
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Eve'],
        'age': [25, 30, np.nan, 35, 40, 40],
        'salary': [50000, 60000, 70000, np.nan, 90000, 90000],
        'department': ['HR', 'IT', 'IT', 'Finance', 'Marketing', 'Marketing'],
        'join_date': ['2020-01-15', '2019-03-20', '2021-07-10', '2018-11-05', '2022-02-28', '2022-02-28']
    }
    
    # Create sample CSV
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_sample_data.csv')
    
    # Validate the cleaned data
    if cleaned_df is not None:
        validate_dataframe(cleaned_df)