import numpy as np

np.set_printoptions(
    precision=2,
    suppress=True,
    formatter={"float": "{: 0.4f}".format},
    edgeitems=57,
    linewidth=1000,
)

x = np.array([5 / 8, 0 / 8, 2 / 8, 4 / 8, 7 / 8, 2 / 8, 3 / 8, 1 / 8], dtype=np.float64)
y = np.fft.ifft(x)
print(x)
print(y*8)
# print(np.abs(y))

# z = np.fft.ifft(y)
# print(np.real(z))
