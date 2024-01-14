import hashlib


def md5(text: str) -> str:
    tokens = text.split(b"\x01")
    tokens = " ".join([t.decode() for t in tokens]).encode('utf-8')
    hashed_tokens = hashlib.md5(tokens)
    return hashed_tokens.hexdigest()
