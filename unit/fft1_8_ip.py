import numpy as np

np.set_printoptions(
    # precision=2,
    suppress=True,
    formatter={"float": "{: 0.6f}".format},
    edgeitems=57,
    linewidth=1000,
)

x = np.array([5 / 8, 0 / 8, 2 / 8, 4 / 8, 7 / 8, 2 / 8, 3 / 8, 1 / 8])  # 5024 7231
print(x)
y = np.fft.fft(x)
print(np.real(y))
print(np.imag(y))

print("----")

ip_dout_re = (
    np.array(
        [
            402653184,
            -92870849,
            117440512,
            25761984,
            167772160,
            25761984,
            117440512,
            -92870848,
        ]
    )
    / 2**27
)
ip_dout_im = (
    np.array(
        [
            0,
            4913932,
            50331648,
            -28640500,
            0,
            28640499,
            -50331648,
            -4913933,
        ]
    )
    / 2**27
)
print(ip_dout_re)
print(ip_dout_im)

print("----")

w = ip_dout_re + ip_dout_im * 1j
v = np.fft.ifft(w)
print(np.real(v))
