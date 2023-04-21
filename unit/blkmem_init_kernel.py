import numpy as np


def float2bytes(value: float, fixed_num: int, int_num: int = 32) -> bytes:
    return int(value * (2**fixed_num)).to_bytes(
        int_num // 8, byteorder="little", signed=True
    )


def GaussKernel(size: tuple[int, int], sigma: float) -> np.ndarray:
    """
    高斯核
    """
    w, h = size
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))  # 根据w, h的值生成一个网格的x，y坐标
    center_x, center_y = (w - 1) / 2, (h - 1) / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5 * dist)
    return labels


with open("kernel.coe", "w") as f:
    f.write("memory_initialization_radix = 16;\n")
    f.write("memory_initialization_vector = \n")
    for x in range(32):
        for y in range(32):
            kernel = np.fft.fft2(GaussKernel(size=(32, 32), sigma=2.0))

            re_bytes = float2bytes(
                value=np.real(kernel[y][x]), fixed_num=16, int_num=32
            )
            im_bytes = float2bytes(
                value=np.imag(kernel[y][x]), fixed_num=16, int_num=32
            )
            f.write(
                f"{im_bytes[3]:02X}{im_bytes[2]:02X}{im_bytes[1]:02X}{im_bytes[0]:02X}"
            )
            f.write(
                f"{re_bytes[3]:02X}{re_bytes[2]:02X}{re_bytes[1]:02X}{re_bytes[0]:02X}"
            )
            if x == 31 and y == 31:
                f.write(f";\n")
            else:
                f.write(f",\n")
