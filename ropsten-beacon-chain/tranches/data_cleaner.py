import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    """
    Load a CSV file, handle missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Display initial info
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(
            lambda col: col.fillna(col.median())
        )
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        print(f"Cleaned shape: {df.shape}")
        print(f"Cleaned data saved to: {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    clean_data("raw_data.csv", "cleaned_data.csv")