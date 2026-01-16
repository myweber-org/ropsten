
import pandas as pd
import hashlib

def remove_duplicates(input_file, output_file, key_columns=None):
    """
    Load a CSV file, remove duplicate rows based on specified columns,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        original_count = len(df)
        
        if key_columns is None:
            key_columns = df.columns.tolist()
        
        df_cleaned = df.drop_duplicates(subset=key_columns, keep='first')
        cleaned_count = len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Original records: {original_count}")
        print(f"Cleaned records: {cleaned_count}")
        print(f"Duplicates removed: {original_count - cleaned_count}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def generate_data_hash(df):
    """
    Generate a hash for the dataframe to verify data integrity.
    """
    data_string = df.to_string(index=False).encode('utf-8')
    return hashlib.md5(data_string).hexdigest()

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = remove_duplicates(input_csv, output_csv)
    
    if cleaned_data is not None:
        data_hash = generate_data_hash(cleaned_data)
        print(f"Data integrity hash: {data_hash}")