import pandas as pd

def remove_duplicates(input_file, output_file, subset_columns=None):
    """
    Load a CSV file, remove duplicate rows, and save the cleaned data.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
        subset_columns (list, optional): List of column names to consider for identifying duplicates.
                                         If None, all columns are considered.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        if subset_columns:
            df_cleaned = df.drop_duplicates(subset=subset_columns)
        else:
            df_cleaned = df.drop_duplicates()
        
        removed_count = initial_count - len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Data cleaning completed:")
        print(f"  Initial rows: {initial_count}")
        print(f"  Removed duplicates: {removed_count}")
        print(f"  Final rows: {len(df_cleaned)}")
        print(f"  Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    columns_to_check = ["id", "email"]
    
    remove_duplicates(input_csv, output_csv, columns_to_check)