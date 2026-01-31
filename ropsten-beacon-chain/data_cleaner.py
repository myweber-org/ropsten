
import pandas as pd
import numpy as np
from pathlib import Path

class CSVDataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded {len(self.df)} rows from {self.file_path.name}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def show_missing_summary(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        
        print("\nMissing Value Summary:")
        print("-" * 40)
        for col in self.df.columns:
            if missing_counts[col] > 0:
                print(f"{col}: {missing_counts[col]} missing ({missing_percent[col]:.1f}%)")
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        df_clean = self.df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                col_type = df_clean[col].dtype
                
                if strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                elif strategy == 'fill' and fill_value is not None:
                    df_clean[col] = df_clean[col].fillna(fill_value)
                elif strategy == 'mean' and np.issubdtype(col_type, np.number):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif strategy == 'median' and np.issubdtype(col_type, np.number):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif strategy == 'mode':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                elif strategy == 'forward':
                    df_clean[col] = df_clean[col].fillna(method='ffill')
                elif strategy == 'backward':
                    df_clean[col] = df_clean[col].fillna(method='bfill')
                else:
                    print(f"Warning: No handling for column {col} with strategy {strategy}")
        
        print(f"Missing values handled using '{strategy}' strategy")
        print(f"Remaining missing values: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        initial_count = len(self.df)
        df_clean = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(df_clean)
        
        print(f"Removed {removed_count} duplicate rows")
        return df_clean
    
    def save_cleaned_data(self, df, output_path=None):
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return output_path

def clean_csv_file(input_file, output_file=None, missing_strategy='mean'):
    cleaner = CSVDataCleaner(input_file)
    
    if not cleaner.load_data():
        return None
    
    cleaner.show_missing_summary()
    cleaned_df = cleaner.handle_missing_values(strategy=missing_strategy)
    cleaned_df = cleaner.remove_duplicates()
    
    if output_file:
        output_path = cleaner.save_cleaned_data(cleaned_df, output_file)
    else:
        output_path = cleaner.save_cleaned_data(cleaned_df)
    
    return cleaned_df, output_path
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': df[column].count()
    }
    
    return stats

def process_numerical_data(df, columns=None):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, processes all numerical columns.
    
    Returns:
    pd.DataFrame: Processed DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    processed_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                processed_df = remove_outliers_iqr(processed_df, column)
            except Exception as e:
                print(f"Error processing column '{column}': {e}")
    
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = process_numerical_data(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))