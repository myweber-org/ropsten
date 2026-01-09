
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a pandas DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    df_clean = df.copy()
    df_clean[column_name] = df_clean[column_name].astype(str)
    df_clean[column_name] = df_clean[column_name].str.strip()
    df_clean[column_name] = df_clean[column_name].str.lower()
    df_clean = df_clean.drop_duplicates(subset=[column_name], keep='first')
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def normalize_text(text):
    """
    Normalize text by removing extra spaces and special characters.
    """
    if not isinstance(text, str):
        return text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def process_file(input_file, output_file, column_to_clean):
    """
    Read a CSV file, clean the specified column, and save to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        df_clean = clean_dataframe(df, column_to_clean)
        df_clean.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    target_column = "product_name"
    
    success = process_file(input_csv, output_csv, target_column)
    if success:
        print("Data cleaning completed successfully.")
    else:
        print("Data cleaning failed.")