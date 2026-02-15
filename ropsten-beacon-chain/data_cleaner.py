import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_na=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    # Standardize column names if mapping is provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Fill missing values with appropriate defaults
    if fill_na:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            elif pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # Convert string columns to lowercase and strip whitespace
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and add validation status.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    
    valid_count = df['email_valid'].sum()
    total_count = len(df)
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a numeric column using the Interquartile Range method.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed = len(df) - len(filtered_df)
    
    print(f"Removed {removed} outliers from column '{column}'")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return filtered_df

def standardize_date_column(df, date_column, output_format='%Y-%m-%d'):
    """
    Standardize date column to a consistent format.
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame")
    
    df = df.copy()
    
    # Try to parse dates with multiple common formats
    date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y.%m.%d', '%d-%m-%Y', '%m-%d-%Y']
    
    for fmt in date_formats:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format=fmt, errors='coerce')
            parsed_count = df[date_column].notna().sum()
            if parsed_count > 0:
                print(f"Parsed {parsed_count} dates with format: {fmt}")
                break
        except:
            continue
    
    # If parsing failed, use pandas' flexible parser
    if df[date_column].isna().all():
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Format to desired output
    df[f"{date_column}_standardized"] = df[date_column].dt.strftime(output_format)
    
    return df

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', None],
        'Email': ['john@example.com', 'jane@example', 'john@example.com', 'bob@test.org', 'invalid'],
        'Age': [25, 30, 25, 150, 35],  # 150 is an outlier
        'Join_Date': ['2023-01-15', '15/01/2023', '2023-01-15', '01-15-2023', '2023.01.20'],
        'Score': [85.5, 92.0, 85.5, 78.3, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    column_map = {'Join_Date': 'join_date', 'Score': 'score'}
    cleaned = clean_dataset(df, column_mapping=column_map)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate emails
    validated = validate_email_column(cleaned, 'Email')
    print("\nEmail Validation Results:")
    print(validated[['Email', 'email_valid']])
    print("\n" + "="*50 + "\n")
    
    # Remove outliers
    filtered = remove_outliers_iqr(cleaned, 'Age')
    print("\nDataFrame after outlier removal:")
    print(filtered)
    print("\n" + "="*50 + "\n")
    
    # Standardize dates
    standardized = standardize_date_column(cleaned, 'join_date')
    print("\nDataFrame with standardized dates:")
    print(standardized[['join_date', 'join_date_standardized']])