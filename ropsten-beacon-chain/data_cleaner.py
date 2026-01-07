
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.columns

        for col in columns:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    fill_value = strategy

                self.df[col] = self.df[col].fillna(fill_value)

        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

        return self

    def standardize_columns(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std > 0:
                self.df[col] = (self.df[col] - mean) / std

        return self

    def get_cleaned_data(self):
        return self.df

    def get_cleaning_report(self):
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]

        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'removed_rows': removed_rows,
            'removed_columns': removed_cols,
            'missing_values_remaining': self.df.isnull().sum().sum()
        }

        return report

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    }

    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)

    cleaned_df = (cleaner
                  .handle_missing_values(strategy='mean')
                  .remove_outliers_iqr(multiplier=1.5)
                  .get_cleaned_data())

    report = cleaner.get_cleaning_report()
    return cleaned_df, report

if __name__ == "__main__":
    result_df, cleaning_report = example_usage()
    print("Cleaned DataFrame:")
    print(result_df)
    print("\nCleaning Report:")
    print(cleaning_report)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using the Interquartile Range method.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | (df_clean[col].isna())]
    return df_clean

def normalize_minmax(df, columns):
    """
    Normalize data using Min-Max scaling.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def normalize_zscore(df, columns):
    """
    Normalize data using Z-score standardization.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    """
    df_filled = df.copy()
    if columns is None:
        columns = df_filled.columns
    
    for col in columns:
        if col in df_filled.columns:
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            elif strategy == 'mode':
                fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = 0
            
            df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def clean_dataset(df, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if outlier_method == 'iqr':
        df = remove_outliers_iqr(df, numeric_cols)
    elif outlier_method == 'zscore':
        df = remove_outliers_zscore(df, numeric_cols)
    
    df = handle_missing_values(df, strategy=missing_strategy, columns=numeric_cols)
    
    if normalize_method == 'minmax':
        df = normalize_minmax(df, numeric_cols)
    elif normalize_method == 'zscore':
        df = normalize_zscore(df, numeric_cols)
    
    return df