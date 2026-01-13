
import os
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password: str, salt: bytes = None):
        self.password = password.encode()
        self.salt = salt if salt else os.urandom(16)
        self.backend = default_backend()
        
    def _derive_key(self, key_length: int = 32) -> bytes:
        kdf = PBKDF2HMAC(
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
        
        output_data = self.salt + iv + ciphertext
        
        with open(output_path, 'wb') as f:
            f.write(output_data)
        
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
    
    @staticmethod
    def generate_key_file(password: str, output_path: str = 'keyfile.bin'):
        salt = os.urandom(16)
        encryptor = FileEncryptor(password, salt)
        key = encryptor._derive_key()
        
        with open(output_path, 'wb') as f:
            f.write(salt + key)
        
        return output_path

def example_usage():
    test_data = b"This is a secret message that needs encryption."
    
    with open('test.txt', 'wb') as f:
        f.write(test_data)
    
    password = "StrongPassword123!"
    encryptor = FileEncryptor(password)
    
    encrypted_file = encryptor.encrypt_file('test.txt')
    print(f"Encrypted file created: {encrypted_file}")
    
    decrypted_file = encryptor.decrypt_file(encrypted_file)
    print(f"Decrypted file created: {decrypted_file}")
    
    with open(decrypted_file, 'rb') as f:
        decrypted_data = f.read()
    
    if test_data == decrypted_data:
        print("Encryption/decryption successful!")
    else:
        print("Encryption/decryption failed!")
    
    os.remove('test.txt')
    os.remove(encrypted_file)
    os.remove(decrypted_file)

if __name__ == '__main__':
    example_usage()
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class SecureFileCrypto:
    def __init__(self, password: str, salt: bytes = None):
        self.password = password.encode()
        self.salt = salt or os.urandom(16)
        self.backend = default_backend()
        
    def _derive_key(self, length: int = 32) -> bytes:
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=length,
            salt=self.salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(self.password)
    
    def encrypt_file(self, input_path: str, output_path: str = None) -> str:
        if not output_path:
            output_path = input_path + '.enc'
        
        key = self._derive_key()
        iv = os.urandom(16)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            f_out.write(self.salt)
            f_out.write(iv)
            
            while True:
                chunk = f_in.read(1024 * 16)
                if not chunk:
                    break
                if len(chunk) % 16 != 0:
                    chunk += b' ' * (16 - len(chunk) % 16)
                f_out.write(encryptor.update(chunk))
            f_out.write(encryptor.finalize())
        
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: str = None) -> str:
        if not output_path:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        with open(input_path, 'rb') as f_in:
            self.salt = f_in.read(16)
            iv = f_in.read(16)
            
            key = self._derive_key()
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            with open(output_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(1024 * 16)
                    if not chunk:
                        break
                    f_out.write(decryptor.update(chunk))
                f_out.write(decryptor.finalize().rstrip(b' '))
        
        return output_path

def create_secure_archive(files: list, password: str, output_name: str = 'archive.enc'):
    import tempfile
    import tarfile
    
    with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp:
        with tarfile.open(tmp.name, 'w') as tar:
            for file_path in files:
                if os.path.exists(file_path):
                    tar.add(file_path)
        
        crypto = SecureFileCrypto(password)
        encrypted_path = crypto.encrypt_file(tmp.name, output_name)
        
        os.unlink(tmp.name)
        return encrypted_path

if __name__ == '__main__':
    test_file = 'test_document.txt'
    with open(test_file, 'w') as f:
        f.write('Sensitive data requiring encryption.\n' * 10)
    
    password = 'StrongPassw0rd!2024'
    crypto = SecureFileCrypto(password)
    
    encrypted = crypto.encrypt_file(test_file)
    print(f'Encrypted: {encrypted}')
    
    decrypted = crypto.decrypt_file(encrypted)
    print(f'Decrypted: {decrypted}')
    
    with open(decrypted, 'r') as f:
        print('Decrypted content preview:', f.read()[:100])
    
    os.remove(test_file)
    os.remove(encrypted)
    os.remove(decrypted)