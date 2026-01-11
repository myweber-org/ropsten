
import os
import sys

def xor_cipher(data, key):
    """Encrypt or decrypt data using XOR cipher."""
    return bytes([b ^ key for b in data])

def process_file(input_path, output_path, key):
    """Encrypt or decrypt a file."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        processed_data = xor_cipher(data, key)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"File processed successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key>")
        print("Key must be an integer between 0 and 255")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        key = int(sys.argv[3])
        if not 0 <= key <= 255:
            raise ValueError("Key must be between 0 and 255")
    except ValueError as e:
        print(f"Invalid key: {e}")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)
    
    if os.path.exists(output_file):
        response = input(f"Output file {output_file} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    success = process_file(input_file, output_file, key)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()