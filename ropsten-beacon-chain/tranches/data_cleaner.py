
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    For numerical columns, missing values are filled with the column median.
    For categorical columns, missing values are filled with the most frequent value.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")

    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows")

    # Handle missing values
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in [np.float64, np.int64]:
            # Numerical column: fill with median
            median_value = df_cleaned[column].median()
            missing_count = df_cleaned[column].isna().sum()
            df_cleaned[column].fillna(median_value, inplace=True)
            if missing_count > 0:
                print(f"Filled {missing_count} missing values in '{column}' with median: {median_value}")
        else:
            # Categorical column: fill with most frequent value
            if df_cleaned[column].isna().any():
                most_frequent = df_cleaned[column].mode()[0]
                missing_count = df_cleaned[column].isna().sum()
                df_cleaned[column].fillna(most_frequent, inplace=True)
                print(f"Filled {missing_count} missing values in '{column}' with most frequent: '{most_frequent}'")

    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    return df_cleaned

def validate_cleaned_data(df):
    """
    Validate that the cleaned DataFrame has no duplicates and no missing values.
    """
    duplicates = df.duplicated().sum()
    missing_values = df.isna().sum().sum()

    if duplicates == 0 and missing_values == 0:
        print("Validation passed: No duplicates and no missing values")
        return True
    else:
        print(f"Validation failed: {duplicates} duplicates, {missing_values} missing values")
        return False

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'age': [25, 30, 30, np.nan, 35, 40],
        'salary': [50000, 60000, 60000, 55000, np.nan, 70000],
        'department': ['HR', 'IT', 'IT', 'Finance', 'HR', np.nan]
    }

    df = pd.DataFrame(sample_data)
    cleaned_df = clean_dataset(df)
    validation_result = validate_cleaned_data(cleaned_df)

    print("\nCleaned DataFrame:")
    print(cleaned_df)