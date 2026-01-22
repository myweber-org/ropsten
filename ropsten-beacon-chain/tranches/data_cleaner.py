
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): Specific columns to clean, if None cleans all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if col in df_clean.columns:
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean.dropna(subset=[col], inplace=True)
    
    return df_clean

def remove_outliers(df, columns=None, threshold=3):
    """
    Remove outliers using z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to check for outliers
    threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64]:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numerical columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64]:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            if std > 0:
                df_clean[col] = (df_clean[col] - mean) / std
    
    return df_clean

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
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_zscore(self, threshold=3):
        df_clean = self.df.copy()
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        return df_clean
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max - col_min > 0:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        return df_normalized
    
    def fill_missing_median(self):
        df_filled = self.df.copy()
        for col in self.numeric_columns:
            median_val = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(median_val)
        return df_filled
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'original_columns': len(self.df.columns),
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().sum(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    data['feature_a'][[10, 25, 50]] = [500, -200, 1000]
    data['feature_b'][[15, 30, 70]] = [np.nan, np.nan, 1000]
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Data Summary:")
    print(cleaner.get_summary())
    
    cleaned_df = cleaner.remove_outliers_zscore()
    filled_df = cleaner.fill_missing_median()
    normalized_df = cleaner.normalize_minmax()
    
    print(f"\nOriginal shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Normalized range - feature_a: [{normalized_df['feature_a'].min():.3f}, {normalized_df['feature_a'].max():.3f}]")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    df = df.dropna()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_dataset(input_file)
        save_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
    except Exception as e:
        print(f"Error during processing: {str(e)}")