
import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

class FileEncryptor:
    def __init__(self, password):
        self.password = password.encode('utf-8')
        self.salt = get_random_bytes(16)
        self.key = self._derive_key()

    def _derive_key(self):
        return PBKDF2(self.password, self.salt, dkLen=32, count=1000000)

    def encrypt_file(self, input_path, output_path):
        iv = get_random_bytes(16)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)

        with open(input_path, 'rb') as f:
            plaintext = f.read()

        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
        with open(output_path, 'wb') as f:
            f.write(self.salt + iv + ciphertext)

    def decrypt_file(self, input_path, output_path):
        with open(input_path, 'rb') as f:
            data = f.read()

        self.salt = data[:16]
        iv = data[16:32]
        ciphertext = data[32:]

        self.key = self._derive_key()
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

        with open(output_path, 'wb') as f:
            f.write(plaintext)

    def calculate_hash(self, file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

def main():
    password = "secure_password_123"
    encryptor = FileEncryptor(password)

    test_file = "test_data.bin"
    encrypted_file = "encrypted.bin"
    decrypted_file = "decrypted.bin"

    with open(test_file, 'wb') as f:
        f.write(get_random_bytes(1024))

    print(f"Original file hash: {encryptor.calculate_hash(test_file)}")
    encryptor.encrypt_file(test_file, encrypted_file)
    print(f"Encrypted file size: {os.path.getsize(encrypted_file)} bytes")
    encryptor.decrypt_file(encrypted_file, decrypted_file)
    print(f"Decrypted file hash: {encryptor.calculate_hash(decrypted_file)}")

    if encryptor.calculate_hash(test_file) == encryptor.calculate_hash(decrypted_file):
        print("Encryption/decryption successful")
    else:
        print("Encryption/decryption failed")

    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)

if __name__ == "__main__":
    main()