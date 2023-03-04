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


def preprocessing(img: np.ndarray, cos_window: np.ndarray, eps=1e-5) -> np.ndarray:
    img = (img - np.mean(img)) / (np.std(img) + eps)
    return cos_window * img


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

        # [保存一些之后用到的变量]

        # 取出第一帧中ground truth的坐标值 x,y为框的左上角坐标，w, h为框的大小
        self.x, self.y, self.w, self.h = bbox
        # 获取框的大小
        self.crop_size = (self.w, self.h)  # 定义裁剪框的大小
        # 定义汉宁窗
        self.cos_window = cosWindow(size=(self.w, self.h))

        # [使用随机变换的groundtruth进行训练得到初始的核]

        # 从第一帧中截取出检测的目标部分
        fi_groundtruth = getSubImage(first_frame, (self.x, self.y, self.w, self.h))
        # 首先生成一个w * h(检测框大小)的高斯核，然后对该高斯核进行傅里叶变换，初始化G
        self._G = np.fft.fft2(gaussianKernel(size=(self.w, self.h), sigma=self.sigma))
        self.Ai = np.zeros_like(self._G)
        self.Bi = np.zeros_like(self._G)
        # 训练初始的核
        fi = fi_groundtruth
        Fi = np.fft.fft2(preprocessing(fi, self.cos_window))
        self.Ai += self._G * np.conj(Fi)
        self.Bi += Fi * np.conj(Fi)
        self.Hi = self.Ai / self.Bi

    def update(self, current_frame, vis=False) -> tuple[int]:
        current_frame = convertImageToFloat(current_frame)

        # [寻找新一帧中的框]

        # 针对当前帧，用前一个目标框的中心截取一个框
        fi = getSubImage(image=current_frame, bbox=(self.x, self.y, self.w, self.h))
        fi = preprocessing(fi, self.cos_window)
        # 卷积得到Gi
        Gi = self.Hi * np.fft.fft2(fi)
        # 对频域下的Gi进行逆傅里叶变换得到实际的gi
        gi = np.real(np.fft.ifft2(Gi))
        if vis is True:
            self.score = gi
        # 获取gi中最大值的index，这个位置就是第二帧图像中目标所在（算法假设物体不会运动超过上个框）
        position = np.unravel_index(np.argmax(gi, axis=None), gi.shape)
        dy, dx = position[0] - (self.h / 2), position[1] - (self.w / 2)
        self.x, self.y = (self.x + dx, self.y + dy)

        # [更新核]

        # 用新的框截出物体
        fi = getSubImage(image=current_frame, bbox=(self.x, self.y, self.w, self.h))
        fi = preprocessing(fi, self.cos_window)
        Fi = np.fft.fft2(fi)
        # 使用学习学习率更新 Hi
        self.Ai = (
            self.interp_factor * (self._G * np.conj(Fi))
            + (1 - self.interp_factor) * self.Ai
        )
        self.Bi = (
            self.interp_factor * (Fi * np.conj(Fi)) + (1 - self.interp_factor) * self.Bi
        )
        self.Hi = self.Ai / self.Bi

        # [返回]

        # 返回新一帧的框 (x, y, w, h)
        return (
            self.x,
            self.y,
            self.w,
            self.h,
        )
