
import argparse
import sys

def remove_duplicates(input_file, output_file):
    seen = set()
    try:
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
        
        with open(output_file, 'w') as outfile:
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    outfile.write(line)
        
        print(f"Successfully removed duplicates. Original lines: {len(lines)}, Unique lines: {len(seen)}")
        return True
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except IOError as e:
        print(f"Error processing files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate lines from a text file.')
    parser.add_argument('input', help='Path to the input file')
    parser.add_argument('output', help='Path to the output file')
    
    args = parser.parse_args()
    
    if not remove_duplicates(args.input, args.output):
        sys.exit(1)

if __name__ == "__main__":
    main()