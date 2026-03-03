
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def calculate_statistics(df):
    stats_dict = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': stats.skew(df[col].dropna())
        }
    return pd.DataFrame(stats_dict).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    
    print("Original dataset shape:", sample_data.shape)
    cleaned_data = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("Cleaned dataset shape:", cleaned_data.shape)
    
    stats_df = calculate_statistics(cleaned_data)
    print("\nStatistical summary:")
    print(stats_df)
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats,
    handling None values, and replacing invalid entries with default.
    """
    cleaned = []
    for val in values:
        if val is None:
            cleaned.append(default)
        elif isinstance(val, str):
            try:
                cleaned.append(float(val))
            except ValueError:
                cleaned.append(default)
        else:
            cleaned.append(float(val))
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    numeric_data = ["1.5", "invalid", None, 3.2, "4.7"]
    print("Numeric data:", numeric_data)
    print("Cleaned numeric:", clean_numeric_data(numeric_data))def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def remove_outliers(self, column, method='iqr', threshold=1.5):
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[column]))
            self.df = self.df[z_scores < threshold]
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = method
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]

def example_usage():
    data = {
        'age': [25, 30, 35, 200, 28, 32, 150, 29, np.nan, 31],
        'income': [50000, 60000, 55000, 1000000, 52000, 58000, 900000, 54000, 56000, 53000],
        'score': [85, 90, 88, 30, 87, 92, 25, 89, 86, 91]
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(df)
    print(f"Original data shape: {cleaner.original_shape}")
    
    cleaner.fill_missing('age', 'mean')
    cleaner.remove_outliers('age', 'iqr', 1.5)
    cleaner.remove_outliers('income', 'zscore', 3)
    cleaner.normalize_column('score', 'minmax')
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Rows removed: {cleaner.get_removed_count()}")
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()