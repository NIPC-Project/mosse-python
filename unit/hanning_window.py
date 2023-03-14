import numpy as np

w = 57
h = 36

result_np = np.hanning(h)[:, np.newaxis].dot(np.hanning(w)[np.newaxis, :])
result_yxj = np.zeros((h, w))

hanning_w = np.zeros(w)
for i in range(w):
    hanning_w[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (w - 1)))
hanning_h = np.zeros(h)
for i in range(h):
    hanning_h[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (h - 1)))

for r in range(h):
    for c in range(w):
        result_yxj[r][c] = hanning_h[r] * hanning_w[c]

# print(result_np)
# print(result_yxj)
# print(result_np - result_yxj)
print(hanning_w)
