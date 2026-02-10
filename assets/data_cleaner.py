
import pandas as pd
import hashlib

def remove_duplicates_by_hash(df, column_name):
    """
    Remove duplicate rows based on hash of specified column.
    """
    seen_hashes = set()
    indices_to_keep = []
    
    for idx, row in df.iterrows():
        content = str(row[column_name]).encode('utf-8')
        content_hash = hashlib.md5(content).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            indices_to_keep.append(idx)
    
    return df.loc[indices_to_keep].reset_index(drop=True)

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    """
    if fill_method == 'mean':
        fill_value = df[column_name].mean()
    elif fill_method == 'median':
        fill_value = df[column_name].median()
    else:
        fill_value = 0
    
    df[column_name] = df[column_name].fillna(fill_value)
    return df

def standardize_text_column(df, column_name):
    """
    Standardize text column by converting to lowercase and stripping whitespace.
    """
    df[column_name] = df[column_name].astype(str).str.lower().str.strip()
    return df

def process_dataframe(input_file, output_file, primary_key_column):
    """
    Main function to process and clean the dataframe.
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        
        df = remove_duplicates_by_hash(df, primary_key_column)
        print(f"After deduplication: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            df = clean_numeric_column(df, col, fill_method='median')
        
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df = standardize_text_column(df, col)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    key_column = "id"
    
    cleaned_df = process_dataframe(input_path, output_path, key_column)