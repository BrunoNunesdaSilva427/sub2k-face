

import time

import serial

from serial_protocol import encode_request, decode_response, RESPONSE_LEN


class ArduinoLink:
    def __init__(self, port, baudrate=115200, timeout=2.0):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2.0)

    def query(self, vector_int8):
        self.ser.reset_input_buffer()
        self.ser.write(encode_request(vector_int8))
        raw = self.ser.read(RESPONSE_LEN)
        if len(raw) < RESPONSE_LEN:
            raise TimeoutError(
                f"Arduino não respondeu a tempo (recebido {len(raw)}/{RESPONSE_LEN} bytes). "
                "Confira a porta serial e se o sketch está com o baudrate certo."
            )
        return decode_response(raw)

    def close(self):
        self.ser.close()
