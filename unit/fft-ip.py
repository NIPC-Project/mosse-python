import numpy as np

np.set_printoptions(
    # precision=2,
    suppress=True,
    formatter={"float": "{: 0.6f}".format},
    edgeitems=57,
    linewidth=1000,
)

x = np.array([5 / 8, 0 / 8, 2 / 8, 4 / 8, 7 / 8, 2 / 8, 3 / 8, 1 / 8])
print(x)
y = np.fft.fft(x)
print(np.real(y))
print(np.imag(y))

print("----")

ip_dout_re = (
    np.array(
        [
            6442450944,
            -1485933569,
            1879048192,
            412191745,
            2684354560,
            412191745,
            1879048192,
            -1485933569,
        ]
    )
    / 2**31
)
ip_dout_im = (
    np.array(
        [
            0,
            78622925,
            805306368,
            -458247988,
            0,
            458247987,
            -805306368,
            -78622924,
        ]
    )
    / 2**31
)
print(ip_dout_re)
print(ip_dout_im)

# print("----")

# w = ip_dout_re + ip_dout_im * 1j
# v = np.fft.ifft(w)
# print(np.real(v))  # 2/8 4/8 7/8 2/8 3/8 1/8 5/8 0/8
