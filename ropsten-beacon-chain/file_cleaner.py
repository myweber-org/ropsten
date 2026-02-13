
import sys
import os

def remove_duplicate_lines(input_file, output_file=None):
    """
    Remove duplicate lines from a text file while preserving order.
    
    Args:
        input_file: Path to the input file
        output_file: Path to the output file (optional, defaults to input_file with '_deduped' suffix)
    
    Returns:
        Number of duplicate lines removed
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found")
    
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_deduped{ext}"
    
    seen_lines = set()
    unique_lines = []
    duplicate_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.rstrip('\n')
            if stripped_line not in seen_lines:
                seen_lines.add(stripped_line)
                unique_lines.append(line)
            else:
                duplicate_count += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)
    
    return duplicate_count

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        removed = remove_duplicate_lines(input_file, output_file)
        output_path = output_file if output_file else f"{os.path.splitext(input_file)[0]}_deduped{os.path.splitext(input_file)[1]}"
        print(f"Removed {removed} duplicate lines")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()