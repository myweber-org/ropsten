
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(cleaned[['feature_a_normalized', 'feature_b_normalized']].head())
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def handle_missing_values(df, column, strategy='mean'):
    """
    Handle missing values in a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if strategy == 'mean':
        fill_value = df_copy[column].mean()
    elif strategy == 'median':
        fill_value = df_copy[column].median()
    elif strategy == 'mode':
        fill_value = df_copy[column].mode()[0] if not df_copy[column].mode().empty else np.nan
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=[column])
        return df_copy
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    df_copy[column] = df_copy[column].fillna(fill_value)
    return df_copy

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10, np.nan, 12, 13, 14, 15]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Remove outliers
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("After outlier removal:")
    print(cleaned_df)
    print()
    
    # Calculate statistics
    stats = calculate_summary_statistics(df, 'values')
    print("Summary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
    
    # Handle missing values
    filled_df = handle_missing_values(df, 'values', strategy='mean')
    print("After handling missing values:")
    print(filled_df)
    print()
    
    # Normalize column
    normalized_df = normalize_column(filled_df, 'values', method='minmax')
    print("After normalization:")
    print(normalized_df)
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(fill_value)
        
        self.df[self.categorical_columns] = self.df[self.categorical_columns].fillna('Unknown')
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self

    def normalize_data(self, method='minmax'):
        if method == 'minmax':
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self

    def get_cleaned_data(self):
        return self.df.copy()

def load_and_clean_data(filepath, cleaning_steps=None):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            if step['action'] == 'handle_missing':
                cleaner.handle_missing_values(**step.get('params', {}))
            elif step['action'] == 'remove_outliers':
                cleaner.remove_outliers_iqr(**step.get('params', {}))
            elif step['action'] == 'normalize':
                cleaner.normalize_data(**step.get('params', {}))
    
    return cleaner.get_cleaned_data()
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
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
        'count': len(df[column]),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def main():
    """
    Example usage of data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(f"Shape: {df.shape}")
    print(f"Summary statistics:")
    print(calculate_summary_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"\nAfter outlier removal:")
    print(f"Shape: {cleaned_df.shape}")
    print(f"Summary statistics:")
    print(calculate_summary_statistics(cleaned_df, 'values'))
    
    normalized_df = normalize_column(cleaned_df, 'values', method='zscore')
    print(f"\nAfter normalization:")
    print(calculate_summary_statistics(normalized_df, 'values'))

if __name__ == "__main__":
    main()