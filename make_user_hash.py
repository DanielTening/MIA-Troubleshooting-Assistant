import json
import hashlib
import secrets

def pbkdf2_hash(password: str, salt: str, iterations: int = 200_000) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return dk.hex()

username = input("Username: ").strip()
password = input("Password: ").strip()

salt = secrets.token_hex(16)
iterations = 200000
password_hash = pbkdf2_hash(password, salt, iterations)

entry = {
    username: {
        "salt": salt,
        "iterations": iterations,
        "password_hash": password_hash
    }
}

print(json.dumps(entry, indent=2))