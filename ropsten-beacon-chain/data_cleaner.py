
import pandas as pd

def clean_data(df):
    """
    Clean the input DataFrame by removing duplicate rows and
    filling missing numeric values with the column mean.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())

    return df_cleaned

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'A': [1, 2, 2, None, 5],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'x']
    })
    print("Original data:")
    print(data)
    cleaned = clean_data(data)
    print("\nCleaned data:")
    print(cleaned)