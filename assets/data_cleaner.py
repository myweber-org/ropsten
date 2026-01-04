
import pandas as pd
import numpy as np
import sys

def clean_csv_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        df.drop_duplicates(inplace=True)
        
        df.to_csv(output_file, index=False)
        
        print(f"Cleaned shape: {df.shape}")
        print(f"Missing values after cleaning:")
        print(df.isnull().sum())
        print(f"Cleaned data saved to: {output_file}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv_data(input_file, output_file)
    
    if success:
        print("Data cleaning completed successfully.")
    else:
        print("Data cleaning failed.")
        sys.exit(1)