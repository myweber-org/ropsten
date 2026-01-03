
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def detect_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        outlier_indices = np.where(z_scores > threshold)[0]
        return self.df[column].dropna().iloc[outlier_indices].index.tolist()
    
    def remove_outliers(self, method='iqr', threshold=1.5):
        outlier_indices = []
        for col in self.numeric_columns:
            if method == 'iqr':
                indices = self.detect_outliers_iqr(col, threshold)
            elif method == 'zscore':
                indices = self.detect_outliers_zscore(col, threshold)
            outlier_indices.extend(indices)
        
        unique_outliers = list(set(outlier_indices))
        cleaned_df = self.df.drop(index=unique_outliers)
        return cleaned_df, len(unique_outliers)
    
    def impute_missing_mean(self):
        imputed_df = self.df.copy()
        for col in self.numeric_columns:
            if imputed_df[col].isnull().any():
                mean_val = imputed_df[col].mean()
                imputed_df[col].fillna(mean_val, inplace=True)
        return imputed_df
    
    def impute_missing_median(self):
        imputed_df = self.df.copy()
        for col in self.numeric_columns:
            if imputed_df[col].isnull().any():
                median_val = imputed_df[col].median()
                imputed_df[col].fillna(median_val, inplace=True)
        return imputed_df
    
    def impute_missing_mode(self):
        imputed_df = self.df.copy()
        for col in self.df.columns:
            if imputed_df[col].isnull().any():
                mode_val = imputed_df[col].mode()[0]
                imputed_df[col].fillna(mode_val, inplace=True)
        return imputed_df
    
    def get_missing_summary(self):
        missing_counts = self.df.isnull().sum()
        missing_percentage = (missing_counts / len(self.df)) * 100
        summary_df = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentage
        })
        return summary_df[summary_df['missing_count'] > 0]
    
    def normalize_data(self, method='minmax'):
        normalized_df = self.df.copy()
        for col in self.numeric_columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        return normalized_df