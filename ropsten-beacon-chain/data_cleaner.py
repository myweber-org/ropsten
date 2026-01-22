import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_cols
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_cols:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_cols
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_cols:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        return normalized_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_cols
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_cols:
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'original_columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_cols),
            'missing_values': self.df[self.numeric_cols].isnull().sum().sum(),
            'data_types': self.df.dtypes.value_counts().to_dict()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(100, 5), 'feature_a'] = np.nan
    
    cleaner = DataCleaner(df)
    print("Data Summary:", cleaner.get_summary())
    
    cleaned = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    normalized = cleaner.normalize_minmax()
    filled = cleaner.fill_missing_median()
    
    return cleaned, normalized, filled

if __name__ == "__main__":
    cleaned_df, normalized_df, filled_df = example_usage()
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Normalized range: [{normalized_df['feature_a'].min():.3f}, {normalized_df['feature_a'].max():.3f}]")
    print(f"Missing values after fill: {filled_df.isnull().sum().sum()}")
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_mapping (dict): Optional dictionary mapping old column names to new ones
    drop_duplicates (bool): Whether to remove duplicate rows
    normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    df_clean = df_clean.replace(['none', 'null', 'nan', ''], np.nan)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def sample_data(df, sample_size=1000, random_state=42):
    """
    Sample data from DataFrame for testing or analysis.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    sample_size (int): Number of rows to sample
    random_state (int): Random seed for reproducibility
    
    Returns:
    pd.DataFrame: Sampled DataFrame
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)