
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save cleaned data. 
                                    If None, overwrites input file.
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_cleaned = df.drop_duplicates()
        final_rows = len(df_cleaned)
        
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
            
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Data cleaning completed:")
        print(f"  Initial rows: {initial_rows}")
        print(f"  Final rows: {final_rows}")
        print(f"  Duplicates removed: {duplicates_removed}")
        print(f"  Cleaned data saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return -1
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)