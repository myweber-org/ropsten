
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    For numerical columns, missing values are filled with the column median.
    For categorical columns, missing values are filled with the most frequent value.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")

    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows")

    # Handle missing values
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in [np.float64, np.int64]:
            # Numerical column: fill with median
            median_value = df_cleaned[column].median()
            missing_count = df_cleaned[column].isna().sum()
            df_cleaned[column].fillna(median_value, inplace=True)
            if missing_count > 0:
                print(f"Filled {missing_count} missing values in '{column}' with median: {median_value}")
        else:
            # Categorical column: fill with most frequent value
            if df_cleaned[column].isna().any():
                most_frequent = df_cleaned[column].mode()[0]
                missing_count = df_cleaned[column].isna().sum()
                df_cleaned[column].fillna(most_frequent, inplace=True)
                print(f"Filled {missing_count} missing values in '{column}' with most frequent: '{most_frequent}'")

    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    return df_cleaned

def validate_cleaned_data(df):
    """
    Validate that the cleaned DataFrame has no duplicates and no missing values.
    """
    duplicates = df.duplicated().sum()
    missing_values = df.isna().sum().sum()

    if duplicates == 0 and missing_values == 0:
        print("Validation passed: No duplicates and no missing values")
        return True
    else:
        print(f"Validation failed: {duplicates} duplicates, {missing_values} missing values")
        return False

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'age': [25, 30, 30, np.nan, 35, 40],
        'salary': [50000, 60000, 60000, 55000, np.nan, 70000],
        'department': ['HR', 'IT', 'IT', 'Finance', 'HR', np.nan]
    }

    df = pd.DataFrame(sample_data)
    cleaned_df = clean_dataset(df)
    validation_result = validate_cleaned_data(cleaned_df)

    print("\nCleaned DataFrame:")
    print(cleaned_df)
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    mean_val = data[column].mean()
    median_val = data[column].median()
    std_val = data[column].std()
    return {
        'mean': mean_val,
        'median': median_val,
        'std': std_val
    }
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
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
        
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                valid_indices = np.where(z_scores < threshold)[0]
                df_clean = df_clean.iloc[valid_indices]
        
        self.df = df_clean
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max != col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        
        self.df = df_normalized
        return self
        
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
                col_mean = df_normalized[col].mean()
                col_std = df_normalized[col].std()
                if col_std > 0:
                    df_normalized[col] = (df_normalized[col] - col_mean) / col_std
        
        self.df = df_normalized
        return self
        
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def summary(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Rows removed: {self.get_removed_count()}")
        print(f"Columns: {list(self.df.columns)}")
        return self

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = np.nan
    
    outlier_indices = np.random.choice(df.index, 20)
    df.loc[outlier_indices, 'feature_a'] = df['feature_a'].mean() + 5 * df['feature_a'].std()
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Sample data created with shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    cleaner.summary()
    
    cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    cleaner.fill_missing_mean()
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaning completed.")
    print("Final shape:", cleaned_df.shape)
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())