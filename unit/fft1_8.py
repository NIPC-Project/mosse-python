import numpy as np

np.set_printoptions(
    precision=2,
    suppress=True,
    formatter={"float": "{: 0.3f}".format},
    edgeitems=57,
    linewidth=1000,
)

x = np.array([0.0, 0, 1, 1, 1, 1, 0, 0])
y = np.fft.fft(x)
print(x)
print(y)
print(np.abs(y))

z = np.fft.ifft(y)
print(np.real(z))
