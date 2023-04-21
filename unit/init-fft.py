import numpy as np

w = 32
h = 32

window = np.array(
    np.hanning(h)[:, np.newaxis].dot(np.hanning(w)[np.newaxis, :])
).reshape(1024)

x, y, w, h = [257, 163, 57, 36]
xc = int(x + w / 2)
yc = int(y + h / 2)

with open("1.bin", "rb") as f:
    test_data = f.read()

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
test_data_f_fft = np.fft.fft2(np.array(test_data_f_mul_window).reshape(32, 32)).reshape(
    1024
)

start = 200
end = 204
print(test_data_f_fft.tolist()[start:end])
