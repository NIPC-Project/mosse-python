import numpy as np


def float2bytes(value: float, fixed_num: int, int_num: int = 32) -> bytes:
    return int(value * (2**fixed_num)).to_bytes(
        int_num // 8, byteorder="little", signed=True
    )


w = 32
h = 32

window = np.hanning(h)[:, np.newaxis].dot(np.hanning(w)[np.newaxis, :])

with open("window.coe", "w") as f:
    f.write("memory_initialization_radix = 16;\n")
    f.write("memory_initialization_vector = \n")
    for x in range(32):
        for y in range(32):
            a_bytes = float2bytes(value=window[y][x], fixed_num=16, int_num=32)
            if x == 31 and y == 31:
                f.write(
                    f"{a_bytes[3]:02X}{a_bytes[2]:02X}{a_bytes[1]:02X}{a_bytes[0]:02X};\n"
                )
            else:
                f.write(
                    f"{a_bytes[3]:02X}{a_bytes[2]:02X}{a_bytes[1]:02X}{a_bytes[0]:02X},\n"
                )
