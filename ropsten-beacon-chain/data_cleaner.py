
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        subset (list): List of column names to consider for duplicates (optional)
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_file)
        
        original_count = len(df)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset, keep='first')
        else:
            df_clean = df.drop_duplicates(keep='first')
        
        removed_count = original_count - len(df_clean)
        
        if output_file:
            df_clean.to_csv(output_file, index=False)
            print(f"Cleaned data saved to: {output_file}")
        
        print(f"Original rows: {original_count}")
        print(f"Removed duplicates: {removed_count}")
        print(f"Remaining rows: {len(df_clean)}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)