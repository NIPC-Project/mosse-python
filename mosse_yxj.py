"""
Python re-implementation of "Visual Object Tracking using Adaptive Correlation Filters"
@inproceedings{Bolme2010Visual,
  title={Visual object tracking using adaptive correlation filters},
  author={Bolme, David S. and Beveridge, J. Ross and Draper, Bruce A. and Lui, Yui Man},
  booktitle={Computer Vision & Pattern Recognition},
  year={2010},
}

Yang Xijie practices at 2023/3.
"""

import numpy as np
import cv2


class BaseCF:
    def __init__(self):
        raise NotImplementedError

    def init(self, first_frame, bbox):
        raise NotImplementedError

    def update(self, current_frame):
        raise NotImplementedError


class MOSSE(BaseCF):
    def __init__(self, ita=0.125, sigma=2.0):
        super(MOSSE).__init__()
        self.ita: float = ita  # 学习率
        self.sigma: sigma = sigma  # 高斯变换中的方差

    def init(self, first_frame: np.ndarray, bbox: tuple[int, int, int, int]):
        first_frame = self._convertImageToFloat(first_frame)
        self.x, self.y, self.w, self.h = bbox

        self.window = self._getHanningWindow(size=(self.w, self.h))
        self.Kernel = np.fft.fft2(
            self._getGaussKernel(size=(self.w, self.h), sigma=self.sigma)
        )

        f = self._getSubImage(first_frame, bbox=(self.x, self.y, self.w, self.h))
        f = self._normalize(f)
        f = self.window * f
        F = np.fft.fft2(f)

        self.A = self.Kernel * np.conj(F)
        self.B = F * np.conj(F)
        self.H = self.A / self.B
        # print(
        #     f"[debug]\nA ({self.A.min()}, {self.A.max()})\nB ({self.B.min()}, {self.B.max()}\nH ({self.H.min()}, {self.H.max()})"
        # )

    def update(self, current_frame, vis=False) -> tuple[int, int, int, int]:
        current_frame = self._convertImageToFloat(current_frame)

        f = self._getSubImage(current_frame, bbox=(self.x, self.y, self.w, self.h))
        f = self._normalize(f)
        f = self.window * f
        F = np.fft.fft2(f)

        G = F * self.H
        g = np.real(np.fft.ifft2(G))

        # debug start
        np.set_printoptions(
            # precision=2,
            suppress=True,
            formatter={"float": "{: .3f}".format},
            edgeitems=10,
            linewidth=1000,
        )
        print(g)
        print(np.max(g))
        exit(0)
        # debug end

        if vis is True:
            self.score = g
        position = np.unravel_index(np.argmax(g, axis=None), g.shape)
        dy, dx = position[0] - ((self.h - 1) / 2), position[1] - ((self.w - 1) / 2)
        self.x, self.y = (self.x + dx, self.y + dy)

        f = self._getSubImage(current_frame, bbox=(self.x, self.y, self.w, self.h))
        f = self._normalize(f)
        f = self.window * f
        F = np.fft.fft2(f)

        self.A = self.ita * (self.Kernel * np.conj(F)) + (1 - self.ita) * self.A
        self.B = self.ita * (F * np.conj(F)) + (1 - self.ita) * self.B
        self.H = self.A / self.B

        return (self.x, self.y, self.w, self.h)

    def _getHanningWindow(self, size: tuple[int, int]) -> np.ndarray:
        """
        汉宁窗，防止频谱泄露
        """
        cos_window = np.hanning(size[1])[:, np.newaxis].dot(
            np.hanning(size[0])[np.newaxis, :]
        )
        return cos_window

    def _getGaussKernel(self, size: tuple[int, int], sigma: float) -> np.ndarray:
        """
        高斯核
        """
        w, h = size
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))  # 根据w, h的值生成一个网格的x，y坐标
        center_x, center_y = (w - 1) / 2, (h - 1) / 2
        dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
        labels = np.exp(-0.5 * dist)
        return labels

    def _convertImageToFloat(self, image: np.ndarray) -> np.ndarray:
        """
        将彩色 8 bit 图像转为灰度 0.0-1.0
        """
        if len(image.shape) != 2:
            assert image.shape[2] == 3
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 归一化
        image = image.astype(np.float32) / 256
        return image

    def _normalize(self, image: np.ndarray, eps=1e-5) -> np.ndarray:
        """
        使得图像数据均值为 0.0  方差为 1.0
        """
        # eps 要结合定点小数的可表示范围来确定
        # 当画面全白或者全黑的时候  不能让处理之后的 image 太大
        # image = (image - np.mean(image)) / (np.std(image) + eps)
        # 硬件实现中  直接移除这个除法  只让均值为 0  方差不管了
        image = image - 0.5
        return image

    def _getSubImage(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        x, y, w, h = bbox
        return cv2.getRectSubPix(image, (w, h), (x + w / 2, y + h / 2))
