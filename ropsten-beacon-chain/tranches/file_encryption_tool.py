
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _process(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str):
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        ciphertext = self._process(plaintext)
        with open(output_path, 'wb') as f:
            f.write(ciphertext)
    
    def decrypt_file(self, input_path: str, output_path: str):
        self.encrypt_file(input_path, output_path)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <key> <input_file> <output_file>")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    key = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    if operation == 'encrypt':
        cipher.encrypt_file(input_file, output_file)
        print(f"File encrypted successfully: {output_file}")
    elif operation == 'decrypt':
        cipher.decrypt_file(input_file, output_file)
        print(f"File decrypted successfully: {output_file}")
    else:
        print("Error: Operation must be 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()