
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    """
    
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        else:
            raise ValueError("Invalid missing_strategy. Use 'mean', 'median', or 'drop'")
        
        print(f"After handling missing values: {df.shape}")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_df is not None:
        print("Data cleaning completed successfully.")
        print(f"Sample of cleaned data:\n{cleaned_df.head()}")