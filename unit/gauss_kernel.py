import numpy as np

w = 57
h = 36

sigma = 2.0

result = np.zeros((h, w))

for r in range(h):
    for c in range(w):
        result[r][c] = np.exp(
            -((c - 0.5 * (w - 1)) ** 2 + (r - 0.5 * (h - 1)) ** 2) / (2 * sigma**2)
        )

print(result.max())
print(result.min())
