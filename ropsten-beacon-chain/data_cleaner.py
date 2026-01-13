
import pandas as pd

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())

    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)

    return df_cleaned

def main():
    # Example usage
    data = {
        'A': [1, 2, 2, None, 5],
        'B': ['x', 'y', None, 'x', 'z'],
        'C': [10.5, None, 10.5, 15.2, 20.1]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df)
    print(cleaned_df)

if __name__ == "__main__":
    main()