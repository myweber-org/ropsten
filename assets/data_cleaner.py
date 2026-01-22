import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean'):
    """
    Load a CSV file and perform basic cleaning operations.
    Handles missing values based on specified strategy.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        if fill_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_strategy == 'drop':
            df.dropna(inplace=True)
        else:
            df.fillna(0, inplace=True)
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values handled: {missing_before} -> {missing_after}")
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to CSV file."""
    if df is not None and not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    else:
        print("No data to save")
        return False

def analyze_data(df):
    """Perform basic analysis on cleaned data."""
    if df is None or df.empty:
        print("No data to analyze")
        return
    
    print("\n=== Data Analysis ===")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, fill_strategy='mean')
    
    if cleaned_df is not None:
        analyze_data(cleaned_df)
        save_cleaned_data(cleaned_df, output_file)