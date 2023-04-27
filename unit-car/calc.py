def bytes2float(value: bytes, fixed_num: int) -> float:
    return int.from_bytes(bytes=value, byteorder="big", signed=True) / (
        2**fixed_num
    )


imag = bytes2float(value=bytes([0x00, 0x00, 0x38, 0xba]), fixed_num=16)
real = bytes2float(value=bytes([0xFF, 0xFF, 0x13, 0x17]), fixed_num=16)
print(f"{real}+{imag}j")
imag = bytes2float(value=bytes([0x00, 0x00, 0x03, 0x7d]), fixed_num=16)
real = bytes2float(value=bytes([0xFF, 0xFe, 0xbf, 0x5c]), fixed_num=16)
print(f"{real}+{imag}j")

imag = bytes2float(value=bytes([0x00, 0x07, 0xff, 0x4b]), fixed_num=16)
real = bytes2float(value=bytes([0x00, 0x17, 0x1b, 0x55]), fixed_num=16)
print(f"{real}+{imag}j")
imag = bytes2float(value=bytes([0xff, 0xfc, 0x76, 0x29]), fixed_num=16)
real = bytes2float(value=bytes([0xFF, 0xef, 0x2b, 0xa7]), fixed_num=16)
print(f"{real}+{imag}j")
