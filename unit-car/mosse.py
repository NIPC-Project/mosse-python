import numpy as np

np.set_printoptions(
    # precision=2,
    suppress=True,
    formatter={"float": "{: .3f}".format},
    edgeitems=10,
    linewidth=1000,
)

# [CONST]

w = 32
h = 32

window = np.array(
    np.hanning(h)[:, np.newaxis].dot(np.hanning(w)[np.newaxis, :])
).reshape(1024)


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


kernel = np.fft.fft2(GaussKernel(size=(32, 32), sigma=2.0))

x, y, w, h = [257, 163, 57, 36]
xc = int(x + w / 2)  # 285
yc = int(y + h / 2)  # 181

# [MOSSE]

with open("1.bin", "rb") as f:
    test_data = f.read()
    # print(test_data[53069])  # 86
    # print(test_data[53070])  # 102
    # print(test_data[53071])  # 135
    # print(test_data[53390])  # 107
    # print(test_data[53391])  # 132

test_data_crop = []
for i in range(32 * 32):
    r = yc - 16 + i // 32
    c = xc - 16 + i % 32
    test_data_crop.append(test_data[r * 320 + c])
test_data_crop_minus_half = []
for i in range(32 * 32):
    a = test_data_crop[i] / 256 - 0.5
    test_data_crop_minus_half.append(a)
test_data_f_mul_window = []
for i in range(32 * 32):
    a = test_data_crop_minus_half[i] * window[i]
    test_data_f_mul_window.append(a)
test_data_f_fft = np.fft.fft2(np.array(test_data_f_mul_window).reshape(32, 32))

start = 0
end = 34
h_ans = test_data_f_fft.reshape(1024)[start:end]
print(h_ans)

a = kernel * np.conj(test_data_f_fft)
b = test_data_f_fft * np.conj(test_data_f_fft)
h = a / b

start = 0
end = 34
h_ans = h.reshape(1024)[start:end]
print(h_ans)
