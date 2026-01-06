import pandas as pd
import numpy as np
import re

def clean_column_names(df):
    """
    Standardize column names: lowercase, replace spaces with underscores, remove special characters.
    """
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)
        col_str = col_str.lower().strip()
        col_str = re.sub(r'\s+', '_', col_str)
        col_str = re.sub(r'[^a-z0-9_]', '', col_str)
        cleaned_columns.append(col_str)
    df.columns = cleaned_columns
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def fill_missing_values(df, numeric_strategy='mean', categorical_strategy='mode'):
    """
    Fill missing values in numeric columns with mean/median and categorical columns with mode.
    """
    df_filled = df.copy()
    
    for column in df_filled.columns:
        if df_filled[column].dtype in ['int64', 'float64']:
            if numeric_strategy == 'mean':
                fill_value = df_filled[column].mean()
            elif numeric_strategy == 'median':
                fill_value = df_filled[column].median()
            else:
                fill_value = 0
            df_filled[column].fillna(fill_value, inplace=True)
        else:
            if categorical_strategy == 'mode' and not df_filled[column].mode().empty:
                fill_value = df_filled[column].mode()[0]
            else:
                fill_value = 'Unknown'
            df_filled[column].fillna(fill_value, inplace=True)
    
    return df_filled

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a numeric column using the Interquartile Range (IQR) method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def standardize_numeric(df, columns):
    """
    Standardize numeric columns to have zero mean and unit variance.
    """
    df_standardized = df.copy()
    for col in columns:
        if col in df_standardized.columns and df_standardized[col].dtype in ['int64', 'float64']:
            mean = df_standardized[col].mean()
            std = df_standardized[col].std()
            if std > 0:
                df_standardized[col] = (df_standardized[col] - mean) / std
    return df_standardized

def clean_csv_file(input_path, output_path, cleaning_steps=None):
    """
    Main function to apply a series of cleaning steps to a CSV file.
    """
    df = pd.read_csv(input_path)
    
    if cleaning_steps is None:
        cleaning_steps = [
            ('clean_column_names', {}),
            ('remove_duplicates', {'subset': None, 'keep': 'first'}),
            ('fill_missing_values', {'numeric_strategy': 'mean', 'categorical_strategy': 'mode'})
        ]
    
    for step, kwargs in cleaning_steps:
        if step == 'clean_column_names':
            df = clean_column_names(df)
        elif step == 'remove_duplicates':
            df = remove_duplicates(df, **kwargs)
        elif step == 'fill_missing_values':
            df = fill_missing_values(df, **kwargs)
        elif step == 'remove_outliers_iqr':
            df = remove_outliers_iqr(df, **kwargs)
        elif step == 'standardize_numeric':
            df = standardize_numeric(df, **kwargs)
    
    df.to_csv(output_path, index=False)
    return dfimport numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        return normalized_df
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        standardized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = standardized_df[col].mean()
                std_val = standardized_df[col].std()
                if std_val > 0:
                    standardized_df[col] = (standardized_df[col] - mean_val) / std_val
        
        return standardized_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
        
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': self.numeric_columns,
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.randint(1, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    data['feature_a'][np.random.choice(100, 5)] = np.nan
    data['feature_b'][np.random.choice(100, 3)] = np.nan
    
    outliers = np.random.choice(100, 2)
    data['feature_a'][outliers] = [500, -200]
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Sample Data Shape:", sample_df.shape)
    print("\nMissing Values:")
    print(sample_df.isnull().sum())
    
    cleaner = DataCleaner(sample_df)
    summary = cleaner.get_summary()
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned_df = cleaner.remove_outliers_iqr()
    print(f"\nAfter outlier removal: {cleaned_df.shape}")
    
    normalized_df = cleaner.normalize_minmax()
    print(f"After normalization: {normalized_df.shape}")
    
    standardized_df = cleaner.standardize_zscore()
    print(f"After standardization: {standardized_df.shape}")
    
    filled_df = cleaner.fill_missing_median()
    print(f"After filling missing values: {filled_df.shape}")
import numpy as np
import pandas as pd

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

def generate_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = generate_sample_data()
    numeric_cols = ['feature_a', 'feature_b']
    result_df = clean_dataset(sample_df, numeric_cols)
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {result_df.shape}")
    print(result_df.head())