

import numpy as np

SYNC_BYTE = 0xAA
VECTOR_LEN = 32
NAME_LEN = 12
RESPONSE_LEN = 1 + NAME_LEN


def encode_request(vector_int8):
    v = np.asarray(vector_int8, dtype=np.int8)
    if v.shape[0] != VECTOR_LEN:
        raise ValueError(f"Vetor precisa ter {VECTOR_LEN} dimensões, recebeu {v.shape[0]}")
    return bytes([SYNC_BYTE]) + v.tobytes()


def decode_response(raw_bytes):
    if len(raw_bytes) != RESPONSE_LEN:
        raise ValueError(
            f"Resposta com tamanho inesperado: {len(raw_bytes)} bytes "
            f"(esperado {RESPONSE_LEN})"
        )
    status = raw_bytes[0]
    if status == 0x00:
        return False, None
    name_bytes = raw_bytes[1:]
    name = name_bytes.split(b"\x00", 1)[0].decode("ascii", errors="replace")
    return True, name


def pad_name(name):
    encoded = name.encode("ascii")[: NAME_LEN - 1]
    return encoded + b"\x00" * (NAME_LEN - len(encoded))
