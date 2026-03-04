
import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file, strategy='mean'):
    try:
        df = pd.read_csv(input_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mode':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
        elif strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return False
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_cleaner.py <input_file> <output_file> [strategy]")
        print("Strategies: mean, median, mode, drop")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'mean'
    
    success = clean_csv(input_file, output_file, strategy)
    sys.exit(0 if success else 1)