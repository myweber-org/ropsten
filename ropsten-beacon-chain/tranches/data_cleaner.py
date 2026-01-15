import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        
    def remove_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
    
    def normalize_minmax(self, column):
        min_val = self.data[column].min()
        max_val = self.data[column].max()
        self.data[column] = (self.data[column] - min_val) / (max_val - min_val)
        return self.data
    
    def fill_missing_mean(self, column):
        mean_val = self.data[column].mean()
        self.data[column].fillna(mean_val, inplace=True)
        return self.data
    
    def clean_all(self, outlier_columns=None, normalize_columns=None, fill_columns=None):
        if outlier_columns:
            for col in outlier_columns:
                self.data = self.remove_outliers_iqr(col)
        
        if normalize_columns:
            for col in normalize_columns:
                self.data = self.normalize_minmax(col)
        
        if fill_columns:
            for col in fill_columns:
                self.data = self.fill_missing_mean(col)
        
        self.cleaned_data = self.data.copy()
        return self.cleaned_data
    
    def save_cleaned_data(self, filename):
        if self.cleaned_data is not None:
            self.cleaned_data.to_csv(filename, index=False)
            return True
        return False

def example_usage():
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 200, 40, 45, None, 50],
        'salary': [50000, 60000, 70000, 800000, 90000, 100000, 110000, 120000],
        'score': [85, 90, 78, 92, 88, None, 95, 87]
    })
    
    cleaner = DataCleaner(sample_data)
    cleaned = cleaner.clean_all(
        outlier_columns=['age', 'salary'],
        normalize_columns=['score'],
        fill_columns=['age', 'score']
    )
    
    print("Original data shape:", sample_data.shape)
    print("Cleaned data shape:", cleaned.shape)
    print("\nCleaned data summary:")
    print(cleaned.describe())
    
    cleaner.save_cleaned_data('cleaned_data.csv')
    return cleaned

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and save the cleaned data.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove rows where all values are NaN
        df = df.dropna(how='all')
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        df.to_csv(output_path, index=False)
        print(f"Data cleaning completed. Cleaned file saved to: {output_path}")
        
        return df, str(output_path)
    
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: The input file is empty")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Perform basic validation on the dataframe.
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    # Check for remaining NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        return False, f"DataFrame contains {nan_count} NaN values"
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            return False, f"Column {col} contains infinite values"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    cleaned_df, output_file = clean_csv_data(input_file)
    
    if cleaned_df is not None:
        is_valid, message = validate_dataframe(cleaned_df)
        print(f"Validation result: {is_valid}")
        print(f"Validation message: {message}")
        
        # Display basic statistics
        print("\nDataFrame Info:")
        print(cleaned_df.info())
        
        print("\nDataFrame Description:")
        print(cleaned_df.describe())