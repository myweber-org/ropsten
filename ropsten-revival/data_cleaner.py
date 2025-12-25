import pandas as pd
import re

def clean_text_column(df, column_name):
    """Standardize text by lowercasing and removing extra whitespace."""
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicates(df, subset_columns):
    """Remove duplicate rows based on specified columns."""
    return df.drop_duplicates(subset=subset_columns, keep='first')

def process_dataframe(input_file, output_file, text_column, key_columns):
    """Main function to load, clean, deduplicate, and save data."""
    df = pd.read_csv(input_file)
    df = clean_text_column(df, text_column)
    df = remove_duplicates(df, key_columns)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    return df

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    text_col = "description"
    key_cols = ["id", "name"]
    process_dataframe(input_path, output_path, text_col, key_cols)