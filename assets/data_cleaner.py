
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    numpy.ndarray: Data with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_stats = {}
        
    def remove_outliers_iqr(self, column, lower_quantile=0.25, upper_quantile=0.75):
        Q1 = self.df[column].quantile(lower_quantile)
        Q3 = self.df[column].quantile(upper_quantile)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.original_stats[column] = {
            'original_count': len(self.df),
            'outliers_removed': ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
        }
        
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        return self
        
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = 0
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df.copy()
        
    def get_cleaning_report(self):
        report = {
            'final_shape': self.df.shape,
            'original_stats': self.original_stats,
            'remaining_columns': list(self.df.columns)
        }
        return report

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(1000, 50), 'feature1'] = np.nan
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                 .remove_outliers_iqr('feature1')
                 .remove_outliers_iqr('feature2')
                 .fill_missing('feature1', strategy='mean')
                 .normalize_column('feature1', method='minmax')
                 .normalize_column('feature2', method='zscore')
                 .get_cleaned_data())
    
    report = cleaner.get_cleaning_report()
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Outliers removed: {report['original_stats']}")
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()