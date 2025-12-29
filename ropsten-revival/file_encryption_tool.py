import os
import sys

def xor_cipher(data: bytes, key: bytes) -> bytes:
    """Encrypt/decrypt data using XOR cipher."""
    return bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])

def process_file(input_path: str, output_path: str, key: str):
    """Encrypt or decrypt a file."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        key_bytes = key.encode('utf-8')
        processed_data = xor_cipher(data, key_bytes)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"File processed successfully: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key>")
        print("Example: python file_encryption_tool.py secret.txt encrypted.txt mypassword")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    process_file(input_file, output_file, key)

if __name__ == "__main__":
    main()