{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "752b2587-4130-4856-8250-316143174e9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import hashlib\n",
    "import secrets\n",
    "\n",
    "def pbkdf2_hash(password: str, salt: str, iterations: int = 200_000) -> str:\n",
    "    dk = hashlib.pbkdf2_hmac(\"sha256\", password.encode(\"utf-8\"), salt.encode(\"utf-8\"), iterations)\n",
    "    return dk.hex()\n",
    "\n",
    "username = input(\"Username: \").strip()\n",
    "password = input(\"Password: \").strip()\n",
    "\n",
    "salt = secrets.token_hex(16)\n",
    "iterations = 200000\n",
    "password_hash = pbkdf2_hash(password, salt, iterations)\n",
    "\n",
    "entry = {\n",
    "    username: {\n",
    "        \"salt\": salt,\n",
    "        \"iterations\": iterations,\n",
    "        \"password_hash\": password_hash\n",
    "    }\n",
    "}\n",
    "\n",
    "print(json.dumps(entry, indent=2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9b9c9aa1-4a14-4662-b296-d9a46295e9ce",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
