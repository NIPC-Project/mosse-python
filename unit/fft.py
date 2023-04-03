import numpy as np
np.set_printoptions(precision=2,suppress=True,formatter={'float': '{: 0.3f}'.format},edgeitems=57,linewidth=1000)

w = 57
h = 36

sigma = 2.0

kernel = np.zeros((h, w))

for r in range(h):
    for c in range(w):
        kernel[r][c] = np.exp(
            -((c - 0.5 * (w - 1)) ** 2 + (r - 0.5 * (h - 1)) ** 2) / (2 * sigma**2)
        )
print(kernel)

Kernel = np.fft.fft2(kernel)
print(Kernel)
