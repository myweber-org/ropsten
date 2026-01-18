
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import re
from datetime import datetime

def clean_dataframe(df):
    """
    Clean a DataFrame by removing duplicates and standardizing date columns.
    """
    # Remove duplicate rows
    initial_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = initial_count - len(df)
    
    # Standardize date columns
    date_pattern = re.compile(r'.*date.*', re.IGNORECASE)
    date_columns = [col for col in df.columns if date_pattern.search(col)]
    
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Fill missing numeric values with column mean
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    return df, removed_duplicates

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV file.
    """
    df.to_csv(output_path, index=False)
    return f"Data exported successfully to {output_path}"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'order_date': ['2023-01-01', '2023-01-01', '2023-02-15', None],
        'customer_id': [101, 101, 102, 103],
        'amount': [150.0, 150.0, 200.0, None],
        'product': ['A', 'A', 'B', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df, duplicates_removed = clean_dataframe(df)
    print(f"\nRemoved {duplicates_removed} duplicate rows")
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Export to file
    result = export_cleaned_data(cleaned_df, 'cleaned_data.csv')
    print(f"\n{result}")