
import pandas as pd
import numpy as np
from pathlib import Path

def clean_dataset(input_path, output_path=None):
    """
    Load a CSV dataset, remove duplicate rows, normalize column names,
    and save the cleaned version.
    """
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")
    
    df_cleaned = df.copy()
    
    df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(' ', '_')
    
    df_cleaned = df_cleaned.drop_duplicates()
    
    df_cleaned = df_cleaned.replace(r'^\s*$', np.nan, regex=True)
    
    cleaned_shape = df_cleaned.shape
    print(f"Cleaned dataset shape: {cleaned_shape}")
    print(f"Removed {original_shape[0] - cleaned_shape[0]} duplicate rows.")
    print(f"Removed {original_shape[1] - cleaned_shape[1]} duplicate columns.")
    
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
    
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    
    return df_cleaned

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    clean_dataset(input_file, output_file)