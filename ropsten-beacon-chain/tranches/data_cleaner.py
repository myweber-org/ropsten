
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
            
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns and pd.api.types.is_numeric_dtype(clean_df[col]):
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - clean_df.shape[0]
        self.df = clean_df
        return removed_count
    
    def normalize_column(self, column, method='zscore'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if method == 'zscore':
            self.df[f'{column}_normalized'] = stats.zscore(self.df[column])
        elif method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        else:
            raise ValueError("Method must be 'zscore' or 'minmax'")
        
        return self.df[f'{column}_normalized']
    
    def fill_missing(self, strategy='mean', custom_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'custom' and custom_value is not None:
                    fill_value = custom_value
                else:
                    continue
                    
                self.df[col].fillna(fill_value, inplace=True)
        
        return self.df.isnull().sum().sum()
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'current_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[indices, 'feature_a'] = np.random.normal(300, 50, 50)
    
    missing_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[missing_indices, 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial shape:", cleaner.original_shape)
    print("Missing values:", sample_df.isnull().sum().sum())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {removed} outliers")
    
    missing_filled = cleaner.fill_missing(strategy='median')
    print(f"Filled {missing_filled} missing values")
    
    cleaner.normalize_column('feature_a', method='zscore')
    cleaner.normalize_column('feature_c', method='minmax')
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"\nFinal shape: {cleaned_df.shape}")
    print("First 5 rows of cleaned data:")
    print(cleaned_df.head())
import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_file, output_file):
    """
    Clean CSV data by handling missing values, converting data types,
    and removing duplicate rows.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Remove rows where critical columns are null
        critical_columns = ['id', 'name', 'value']
        existing_critical = [col for col in critical_columns if col in df.columns]
        if existing_critical:
            df = df.dropna(subset=existing_critical)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning completed. Cleaned file saved to {output_file}")
        print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, rules):
    """
    Validate data against predefined rules.
    """
    validation_results = {}
    
    for column, rule in rules.items():
        if column in df.columns:
            if rule['type'] == 'numeric':
                min_val = rule.get('min', -np.inf)
                max_val = rule.get('max', np.inf)
                invalid_count = ((df[column] < min_val) | (df[column] > max_val)).sum()
                validation_results[column] = {
                    'valid': invalid_count == 0,
                    'invalid_count': invalid_count
                }
            elif rule['type'] == 'categorical':
                allowed_values = rule.get('allowed_values', [])
                if allowed_values:
                    invalid_count = (~df[column].isin(allowed_values)).sum()
                    validation_results[column] = {
                        'valid': invalid_count == 0,
                        'invalid_count': invalid_count
                    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    # Define validation rules
    validation_rules = {
        'age': {'type': 'numeric', 'min': 0, 'max': 120},
        'score': {'type': 'numeric', 'min': 0, 'max': 100},
        'status': {'type': 'categorical', 'allowed_values': ['active', 'inactive', 'pending']}
    }
    
    cleaned_df = clean_csv_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        validation_results = validate_data(cleaned_df, validation_rules)
        print("Validation Results:")
        for column, result in validation_results.items():
            print(f"{column}: Valid={result['valid']}, Invalid={result['invalid_count']}")