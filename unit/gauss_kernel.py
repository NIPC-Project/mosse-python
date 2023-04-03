import numpy as np

w = 64
h = 32

sigma = 2.0

result = np.zeros((h, w))

for r in range(h):
    for c in range(w):
        result[r][c] = np.exp(
            -((c - 0.5 * (w - 1)) ** 2 + (r - 0.5 * (h - 1)) ** 2) / (2 * sigma**2)
        )

print(result)
print(result.max())
print(result.min())


def _getGaussKernel(size: tuple[int, int], sigma: float) -> np.ndarray:
    """
    高斯核
    """
    w, h = size
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))  # 根据w, h的值生成一个网格的x，y坐标
    center_x, center_y = (w - 1) / 2, (h - 1) / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5 * dist)
    return labels


kernel = _getGaussKernel(size=(64, 32), sigma=2.0)
print(kernel)
