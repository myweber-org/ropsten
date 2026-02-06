import pandas as pd

def clean_dataset(input_file, output_file):
    """
    Load a CSV file, remove rows with null values,
    drop duplicate rows, and save the cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = df.shape[0]
        
        df_cleaned = df.dropna()
        df_cleaned = df_cleaned.drop_duplicates()
        final_rows = df_cleaned.shape[0]
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Data cleaning completed.")
        print(f"Initial rows: {initial_rows}")
        print(f"Rows after cleaning: {final_rows}")
        print(f"Removed rows: {initial_rows - final_rows}")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: The input file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    clean_dataset(input_csv, output_csv)