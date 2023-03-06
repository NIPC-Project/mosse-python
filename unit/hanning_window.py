import numpy as np

N = 7

result_np = np.hanning(N)
result_yxj = np.zeros_like(result_np)

for i in range(N):
    result_yxj[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (N - 1)))

print(result_np)
print(result_yxj)
