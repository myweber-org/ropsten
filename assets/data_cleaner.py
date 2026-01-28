
import pandas as pd
import re

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Define a helper function for string normalization
    def normalize_string(text):
        if pd.isna(text):
            return text
        # Convert to string, strip whitespace, and convert to lowercase
        text = str(text).strip().lower()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text
    
    # Apply normalization to specified columns
    for col in column_names:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(normalize_string)
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

# Example usage (commented out for library use)
# if __name__ == "__main__":
#     # Load your dataset
#     data = pd.read_csv('raw_data.csv')
#     # Specify columns to normalize (e.g., 'name', 'email')
#     columns_to_clean = ['name', 'email', 'address']
#     cleaned_data = clean_dataset(data, columns_to_clean)
#     save_cleaned_data(cleaned_data, 'cleaned_data.csv')