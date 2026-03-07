
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
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
        
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'cleaned_rows': len(self.df),
            'columns': self.original_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    data['feature1'][:50] = np.nan
    data['feature2'][[10, 20, 30]] = 1000
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    cleaner.fill_missing('feature1', 'mean')
    cleaner.remove_outliers_iqr('feature2')
    cleaner.normalize_column('feature1', 'standard')
    cleaner.normalize_column('feature2', 'minmax')
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    return cleaned_df, summary

if __name__ == "__main__":
    cleaned_data, stats_summary = example_usage()
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Summary stats: {stats_summary}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return dataframe[(dataframe[column] >= lower_bound) & 
                     (dataframe[column] <= upper_bound)]

def z_score_normalize(dataframe, column):
    """
    Normalize column using z-score normalization
    """
    mean = dataframe[column].mean()
    std = dataframe[column].std()
    
    if std == 0:
        return dataframe[column]
    
    return (dataframe[column] - mean) / std

def min_max_normalize(dataframe, column):
    """
    Normalize column using min-max scaling
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val == min_val:
        return dataframe[column]
    
    return (dataframe[column] - min_val) / (max_val - min_val)

def clean_dataset(dataframe, numeric_columns, outlier_threshold=1.5, normalization_method='zscore'):
    """
    Clean dataset by removing outliers and normalizing numeric columns
    """
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            if normalization_method == 'zscore':
                cleaned_df[column] = z_score_normalize(cleaned_df, column)
            elif normalization_method == 'minmax':
                cleaned_df[column] = min_max_normalize(cleaned_df, column)
    
    return cleaned_df

def detect_skewed_columns(dataframe, numeric_columns, skew_threshold=0.5):
    """
    Detect columns with significant skewness
    """
    skewed_cols = []
    
    for column in numeric_columns:
        if column in dataframe.columns:
            skewness = dataframe[column].skew()
            if abs(skewness) > skew_threshold:
                skewed_cols.append((column, skewness))
    
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

def log_transform_skewed(dataframe, skewed_columns):
    """
    Apply log transformation to reduce skewness
    """
    transformed_df = dataframe.copy()
    
    for column, _ in skewed_columns:
        if column in transformed_df.columns:
            min_val = transformed_df[column].min()
            if min_val <= 0:
                transformed_df[column] = np.log(transformed_df[column] - min_val + 1)
            else:
                transformed_df[column] = np.log(transformed_df[column])
    
    return transformed_df