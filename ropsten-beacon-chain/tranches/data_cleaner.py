import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalize(data, column):
    """
    Normalize a DataFrame column using Z-score normalization.
    """
    mean = data[column].mean()
    std = data[column].std()
    data[column + '_normalized'] = (data[column] - mean) / std
    return data

def min_max_normalize(data, column):
    """
    Normalize a DataFrame column using Min-Max scaling to range [0, 1].
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_scaled'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization_method='zscore'):
    """
    Main function to clean dataset by removing outliers and applying normalization.
    """
    cleaned_df = df.copy()
    
    if outlier_removal:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalization_method == 'zscore':
                cleaned_df = z_score_normalize(cleaned_df, col)
            elif normalization_method == 'minmax':
                cleaned_df = min_max_normalize(cleaned_df, col)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    print("Original dataset shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2'], outlier_removal=True, normalization_method='zscore')
    print("Cleaned dataset shape:", cleaned.shape)
    print("Cleaned dataset columns:", cleaned.columns.tolist())
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def z_score_normalization(df, column):
    """Apply z-score normalization to a column."""
    df[column + '_normalized'] = stats.zscore(df[column])
    return df

def min_max_normalization(df, column):
    """Apply min-max normalization to a column."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_scaled'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    """Main cleaning pipeline."""
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = z_score_normalization(df, col)
        df = min_max_normalization(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
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
    return {'mean': mean_val, 'median': median_val, 'std': std_val}
import pandas as pd

def clean_dataset(df):
    """
    Clean the dataset by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)
    
    return df_cleaned

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 90.0, 90.0, 78.5, None, 95.0]
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df)
    print("Cleaned dataset:")
    print(cleaned_df)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def normalize_column(self, column: str, method: str = 'minmax') -> pd.DataFrame:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if method == 'minmax':
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[f"{column}_normalized"] = (self.df[column] - col_min) / (col_max - col_min)
            else:
                self.df[f"{column}_normalized"] = 0.5
                
        elif method == 'zscore':
            col_mean = self.df[column].mean()
            col_std = self.df[column].std()
            if col_std > 0:
                self.df[f"{column}_normalized"] = (self.df[column] - col_mean) / col_std
            else:
                self.df[f"{column}_normalized"] = 0
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return self.df
    
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col not in self.df.columns:
                continue
                
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            if strategy == 'mean':
                fill_value = self.df[col].mean()
            elif strategy == 'median':
                fill_value = self.df[col].median()
            elif strategy == 'mode':
                fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
                print(f"Dropped {missing_count} rows with missing values in column '{col}'")
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            self.df[col] = self.df[col].fillna(fill_value)
            print(f"Filled {missing_count} missing values in column '{col}' with {strategy}: {fill_value}")
            
        return self.df
    
    def get_summary(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'current_rows': len(self.df),
            'original_columns': self.original_shape[1],
            'current_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates_removed': self.original_shape[0] - len(self.df)
        }

def clean_dataset(file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy='median')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:3]:
        cleaner.normalize_column(col, method='minmax')
    
    summary = cleaner.get_summary()
    print("Cleaning summary:", summary)
    
    if output_path:
        cleaner.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return cleaner.df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 5, 5],
        'value': [10, 20, 20, np.nan, 40, 50, 50],
        'score': [1.5, 2.3, 2.3, 3.1, 4.0, 5.5, 5.5]
    })
    
    cleaner = DataCleaner(sample_data)
    print("Original data shape:", cleaner.original_shape)
    
    cleaner.remove_duplicates(subset=['id'])
    cleaner.handle_missing_values(strategy='mean', columns=['value'])
    cleaner.normalize_column('score', method='minmax')
    
    print("Cleaned data shape:", cleaner.df.shape)
    print("Cleaned data:")
    print(cleaner.df)