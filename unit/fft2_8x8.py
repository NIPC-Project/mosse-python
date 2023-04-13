import numpy as np

np.set_printoptions(
    precision=2,
    suppress=True,
    formatter={"float": "{: 0.1f}".format},
    edgeitems=57,
    linewidth=1000,
)

a = np.array(range(64), dtype=np.float64).reshape((8, 8))
print(a)

f_ans = np.fft.fft2(a)

f_1 = np.zeros_like(a, dtype=np.complex64)
for i in range(8):
    f_1[i, :] = np.fft.fft(a[i, :])

f_2 = np.zeros_like(a, dtype=np.complex64)
for i in range(8):
    f_2[:, i] = np.fft.fft(f_1[:, i])

print(f_2 - f_ans)
