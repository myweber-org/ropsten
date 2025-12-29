import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str): Path for cleaned output file (optional)
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        original_rows = len(df)
        
        df = df.drop_duplicates()
        print(f"Removed {original_rows - len(df)} duplicate rows")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
                    print(f"Filled missing values in {col} with mean: {mean_val:.2f}")
        
        elif missing_strategy == 'median':
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"Filled missing values in {col} with median: {median_val:.2f}")
        
        elif missing_strategy == 'drop':
            df = df.dropna()
            print(f"Dropped rows with missing values")
        
        else:
            print(f"Unknown strategy: {missing_strategy}. Using 'mean' as default.")
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        print(f"Final data shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if df is None or df.empty:
        print("Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    print(f"Data validation passed. Shape: {df.shape}, Columns: {list(df.columns)}")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'value': [10.5, None, 15.2, 10.5, None, 18.7],
        'category': ['A', 'B', 'A', 'A', 'B', 'C']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Validation result: {validation_result}")
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): Specific columns to clean, if None clean all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() == 0:
            continue
        
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        elif strategy == 'mode':
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col].fillna(mode_val[0], inplace=True)
        
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to check for outliers
    threshold (float): IQR multiplier threshold
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue
        
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        if col not in df_standardized.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(df_standardized[col]):
            continue
        
        mean_val = df_standardized[col].mean()
        std_val = df_standardized[col].std()
        
        if std_val > 0:
            df_standardized[col] = (df_standardized[col] - mean_val) / std_val
    
    return df_standardized

def clean_csv_file(input_path, output_path, missing_strategy='mean', remove_outliers=True):
    """
    Complete pipeline for cleaning a CSV file.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str): Path to save cleaned CSV file
    missing_strategy (str): Strategy for handling missing values
    remove_outliers (bool): Whether to remove outliers
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        df = pd.read_csv(input_path)
        
        df_clean = clean_missing_values(df, strategy=missing_strategy)
        
        if remove_outliers:
            df_clean = remove_outliers_iqr(df_clean)
        
        df_clean = standardize_columns(df_clean)
        
        df_clean.to_csv(output_path, index=False)
        
        print(f"Data cleaning completed. Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
        print(f"Cleaned data saved to: {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_sample)
    print("\nCleaned DataFrame (mean imputation):")
    df_clean = clean_missing_values(df_sample, strategy='mean')
    print(df_clean)import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.columns

        for col in columns:
            if self.df[col].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                self.df[col].fillna(fill_value, inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            columns = list(numeric_cols)

        for col in columns:
            if self.df[col].dtype in ['int64', 'float64']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def get_cleaned_data(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        return self.df

    def save_cleaned_data(self, filepath):
        self.df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', np.nan, 'y', 'x']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)

    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .handle_missing_values(strategy='mean')
                  .remove_outliers_iqr(multiplier=1.5)
                  .get_cleaned_data())
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("\nCleaned DataFrame:")
    print(result)