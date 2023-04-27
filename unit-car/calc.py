def bytes2float(value: bytes, fixed_num: int) -> float:
    return int.from_bytes(bytes=value, byteorder="big", signed=True) / (2**fixed_num)


# [F - 0 & 33]
imag = bytes2float(value=bytes([0x00, 0x07, 0xFF, 0x4B]), fixed_num=16)
real = bytes2float(value=bytes([0x00, 0x17, 0x1B, 0x55]), fixed_num=16)
print(f"{real}+{imag}j")
imag = bytes2float(value=bytes([0xFF, 0xFC, 0x76, 0x29]), fixed_num=16)
real = bytes2float(value=bytes([0xFF, 0xEF, 0x2B, 0xA7]), fixed_num=16)
print(f"{real}+{imag}j")

# [H - 0 & 33]
imag = bytes2float(value=bytes([0x00, 0x00, 0x38, 0xBA]), fixed_num=16)
real = bytes2float(value=bytes([0xFF, 0xFF, 0x13, 0x17]), fixed_num=16)
print(f"{real}+{imag}j")
imag = bytes2float(value=bytes([0x00, 0x00, 0x03, 0x7D]), fixed_num=16)
real = bytes2float(value=bytes([0xFF, 0xFE, 0xBF, 0x5C]), fixed_num=16)
print(f"{real}+{imag}j")

# [G - 0 & 33]
imag = bytes2float(value=bytes([0x00, 0x19, 0x22, 0x1D]), fixed_num=16)
real = bytes2float(value=bytes([0x00, 0x00, 0x00, 0x00]), fixed_num=16)
print(f"{real}+{imag}j")
imag = bytes2float(value=bytes([0x00, 0x15, 0x20, 0x8F]), fixed_num=16)
real = bytes2float(value=bytes([0x00, 0x04, 0x33, 0xDC]), fixed_num=16)
print(f"{real}+{imag}j")
