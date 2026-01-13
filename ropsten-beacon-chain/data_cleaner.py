
import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_path, output_path):
    """
    Load, clean, and save CSV data by handling missing values,
    removing duplicates, and standardizing date formats.
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
        
        # Standardize date columns
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.strftime('%Y-%m-%d')
            except:
                continue
        
        # Remove rows where critical columns are null
        critical_columns = ['id', 'name']
        existing_critical = [col for col in critical_columns if col in df.columns]
        if existing_critical:
            df = df.dropna(subset=existing_critical)
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Data cleaning complete. Cleaned data saved to: {output_path}")
        print(f"Original rows: {len(pd.read_csv(input_path))}, Cleaned rows: {len(df)}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df):
    """
    Perform basic data validation checks.
    """
    if df is None or df.empty:
        return False
    
    validation_results = {
        'has_duplicates': df.duplicated().any(),
        'missing_values': df.isnull().sum().sum(),
        'numeric_range_issues': 0
    }
    
    # Check for numeric values in expected range
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].min() < 0 and 'age' in col.lower():
            validation_results['numeric_range_issues'] += 1
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        validation = validate_data(cleaned_df)
        print("Validation results:", validation)