import numpy as np
import cv2


def cosWindow(size: tuple[int, int]) -> np.ndarray:
    """
    汉宁窗，防止频谱泄露
    """
    cos_window = np.hanning(int(size[1]))[:, np.newaxis].dot(
        np.hanning(int(size[0]))[np.newaxis, :]
    )
    return cos_window


def gaussianKernel(size: tuple[int, int], sigma: float) -> np.ndarray:
    """
    高斯核
    """
    w, h = size
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))  # 根据w, h的值生成一个网格的x，y坐标
    center_x, center_y = w / 2, h / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5 * dist)
    return labels


def convertImageToFloat(image: np.ndarray) -> np.ndarray:
    """
    将彩色 8 bit 图像转为灰度 0.0-1.0
    """
    if len(image.shape) != 2:
        assert image.shape[2] == 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 归一化
    image = image.astype(np.float32) / 255
    return image


def normalize(image: np.ndarray, eps=1e-5) -> np.ndarray:
    """
    使得图像数据均值为 0.0  方差为 1.0
    """
    image = (image - np.mean(image)) / (np.std(image) + eps)
    return image


def getSubImage(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return cv2.getRectSubPix(image, (w, h), (x + w / 2, y + h / 2))


class BaseCF:
    def __init__(self):
        raise NotImplementedError

    def init(self, first_frame, bbox):
        raise NotImplementedError

    def update(self, current_frame):
        raise NotImplementedError


class MOSSE(BaseCF):
    def __init__(self, interp_factor=0.125, sigma=2.0):
        super(MOSSE).__init__()
        self.interp_factor: float = interp_factor  # 学习率
        self.sigma: sigma = sigma  # 高斯变换中的方差

    def init(self, first_frame: np.ndarray, bbox: tuple[int, int, int, int]):
        first_frame = convertImageToFloat(image=first_frame)
        self.x, self.y, self.w, self.h = bbox

        self.window = cosWindow(size=(self.w, self.h))
        self.Kernel = np.fft.fft2(
            gaussianKernel(size=(self.w, self.h), sigma=self.sigma)
        )

        f = getSubImage(first_frame, (self.x, self.y, self.w, self.h))
        f = normalize(f)
        f = self.window * f

        F = np.fft.fft2(f)

        self.A = self.Kernel * np.conj(F)
        self.B = F * np.conj(F)
        self.H = self.A / self.B

    def update(self, current_frame, vis=False) -> tuple[int]:
        current_frame = convertImageToFloat(current_frame)
        f = getSubImage(image=current_frame, bbox=(self.x, self.y, self.w, self.h))
        f = normalize(f)
        f = self.window * f
        F = np.fft.fft2(f)

        G = self.H * F
        g = np.real(np.fft.ifft2(G))
        if vis is True:
            self.score = g
        position = np.unravel_index(np.argmax(g, axis=None), g.shape)
        dy, dx = position[0] - (self.h / 2), position[1] - (self.w / 2)
        self.x, self.y = (self.x + dx, self.y + dy)

        # [更新核]

        f = getSubImage(image=current_frame, bbox=(self.x, self.y, self.w, self.h))
        f = normalize(f)
        f = self.window * f
        F = np.fft.fft2(f)

        self.A = (
            self.interp_factor * (self.Kernel * np.conj(F))
            + (1 - self.interp_factor) * self.A
        )
        self.B = (
            self.interp_factor * (F * np.conj(F)) + (1 - self.interp_factor) * self.B
        )
        self.H = self.A / self.B

        # [返回]

        # 返回新一帧的框 (x, y, w, h)
        return (
            self.x,
            self.y,
            self.w,
            self.h,
        )
