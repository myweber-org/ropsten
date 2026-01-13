import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Clean a CSV file by removing duplicates, handling missing values,
    and standardizing column names.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        # Remove rows where critical columns are null
        critical_columns = ['id', 'name', 'date']
        existing_critical = [col for col in critical_columns if col in df.columns]
        if existing_critical:
            df = df.dropna(subset=existing_critical)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaned successfully. Output saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_csv_data(input_file, output_file)