import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password: str, salt: bytes = None):
        self.password = password.encode()
        self.salt = salt or os.urandom(16)
        self.backend = default_backend()
        
    def _derive_key(self, key_length: int = 32) -> bytes:
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=self.salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(self.password)
    
    def encrypt_file(self, input_path: str, output_path: str = None) -> str:
        if not output_path:
            output_path = input_path + '.enc'
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        key = self._derive_key()
        iv = os.urandom(16)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        pad_length = 16 - (len(plaintext) % 16)
        padded_data = plaintext + bytes([pad_length] * pad_length)
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        result = self.salt + iv + ciphertext
        
        with open(output_path, 'wb') as f:
            f.write(result)
        
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: str = None) -> str:
        if not output_path:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        self.salt = data[:16]
        iv = data[16:32]
        ciphertext = data[32:]
        
        key = self._derive_key()
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        pad_length = padded_plaintext[-1]
        plaintext = padded_plaintext[:-pad_length]
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path

def main():
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <password> [output_file]")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    password = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    encryptor = FileEncryptor(password)
    
    try:
        if operation == 'encrypt':
            result = encryptor.encrypt_file(input_file, output_file)
            print(f"File encrypted successfully: {result}")
        elif operation == 'decrypt':
            result = encryptor.decrypt_file(input_file, output_file)
            print(f"File decrypted successfully: {result}")
        else:
            print("Invalid operation. Use 'encrypt' or 'decrypt'.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _process(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            ciphertext = self._process(plaintext)
            with open(output_path, 'wb') as f:
                f.write(ciphertext)
            return True
        except Exception as e:
            print(f"Encryption error: {e}", file=sys.stderr)
            return False
    
    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        return self.encrypt_file(input_path, output_path)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <key> <input_file> <output_file>")
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
        success = cipher.encrypt_file(input_file, output_file)
        if success:
            print(f"File encrypted successfully: {output_file}")
        else:
            print("Encryption failed.")
    elif operation == 'decrypt':
        success = cipher.decrypt_file(input_file, output_file)
        if success:
            print(f"File decrypted successfully: {output_file}")
        else:
            print("Decryption failed.")
    else:
        print("Error: Operation must be 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()