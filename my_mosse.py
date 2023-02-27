"""
Python re-implementation of "Visual Object Tracking using Adaptive Correlation Filters"
@inproceedings{Bolme2010Visual,
  title={Visual object tracking using adaptive correlation filters},
  author={Bolme, David S. and Beveridge, J. Ross and Draper, Bruce A. and Lui, Yui Man},
  booktitle={Computer Vision & Pattern Recognition},
  year={2010},
}

Dongmen practices at 2020/3/26.
"""

from typing import Tuple
import numpy as np
import cv2


# 汉宁窗，防止频谱泄露
def cos_window(sz):
    """
    width, height = sz
    j = np.arange(0, width)
    i = np.arange(0, height)
    J, I = np.meshgrid(j, i)
    cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
    """

    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(
        np.hanning(int(sz[0]))[np.newaxis, :]
    )
    return cos_window


def gaussian2d_labels(sz, sigma):
    """
    该函数的作用是生成一个sz大小的的高斯核
    """
    w, h = sz
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))  # 根据w, h的值生成一个网格的x，y坐标
    center_x, center_y = w / 2, h / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5 * dist)
    return labels


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
        self.interp_factor = interp_factor  # 学习率
        self.sigma = sigma  # 高斯变换中的方差

    # first_frame: np.array
    # bbox: x y w h
    def init(self, first_frame: np.ndarray, bbox: Tuple[int]):
        if len(first_frame.shape) != 2:
            assert first_frame.shape[2] == 3
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)  # RGB图片转换成灰度图片
        first_frame = first_frame.astype(np.float32) / 255  # 归一化
        x, y, w, h = bbox  # 取出第一帧中ground truth的坐标值 x,y为框的左上角坐标，w, h为框的大小
        self.center = (x + w / 2, y + h / 2)  # 计算ground truth的中心
        self.w, self.h = w, h  # 获取框的大小
        w, h = int(round(w)), int(round(h))  # round()四舍五入
        self.cos_window = cos_window((w, h))  # 定义汉宁窗
        # 从第一帧中截取出检测的目标部分， _fi.shape = (w,h)
        self._fi = cv2.getRectSubPix(
            image=first_frame, patchSize=(w, h), center=self.center
        )
        # 首先生成一个w * h(检测框大小)的高斯核，然后对该高斯核进行傅里叶变换，初始化G
        self._G = np.fft.fft2(gaussian2d_labels((w, h), self.sigma))
        self.crop_size = (w, h)  # 定义裁剪框的大小
        self.Ai = np.zeros_like(self._G)  # 初始化Ai
        self.Bi = np.zeros_like(self._G)  # 初始化Bi
        # 对Fi进行多次刚性形变，增强检测的鲁棒性
        # 计算出Ai和Bi的初始值
        for _ in range(8):
            fi = self.randomWarp(self._fi)  # YXJ: 如果测试场景不发生大小变化、旋转等，不需要做
            # fi = self._fi
            Fi = np.fft.fft2(self.preprocessing(fi, self.cos_window))
            self.Ai += self._G * np.conj(Fi)
            self.Bi += Fi * np.conj(Fi)
        self.Hi = self.Ai / self.Bi

    def update(self, current_frame, vis=False) -> tuple[int]:
        # [获取新的一帧]

        if len(current_frame.shape) != 2:
            assert current_frame.shape[2] == 3
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_frame = current_frame.astype(np.float32) / 255

        # [寻找新一帧中的框]

        # 针对当前帧，用前一个目标框的中心截取一个框
        fi = cv2.getRectSubPix(
            current_frame, (int(round(self.w)), int(round(self.h))), self.center
        )
        fi = self.preprocessing(fi, self.cos_window)
        # 卷积得到Gi
        Gi = self.Hi * np.fft.fft2(fi)
        # 对频域下的Gi进行逆傅里叶变换得到实际的gi
        gi = np.real(np.fft.ifft2(Gi))  
        if vis is True:
            self.score = gi
        # 获取gi中最大值的index，这个位置就是第二帧图像中目标所在（算法假设物体不会运动超过上个框）
        position = np.unravel_index(np.argmax(gi, axis=None), gi.shape)
        dy, dx = position[0] - (self.h / 2), position[1] - (self.w / 2)
        x_c, y_c = self.center
        self.center = (x_c + dx, y_c + dy)  

        # [更新核]

        # 用新的框截出物体
        fi = cv2.getRectSubPix(
            current_frame, (int(round(self.w)), int(round(self.h))), self.center
        )
        fi = self.preprocessing(fi, self.cos_window)
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
            self.center[0] - self.w / 2,
            self.center[1] - self.h / 2,
            self.w,
            self.h,
        )

    def preprocessing(
        self, img: np.ndarray, cos_window: np.ndarray, eps=1e-5
    ) -> np.ndarray:
        """
        该函数对数据进行预处理：
        1. 对数据矩阵取对数
        2. 接着标准化数据，使其更加符合标准正态分布
        (经过以上两步处理，直观上来说数据变得中心化了，弱化了其背景的影响)
        3. 使用窗函数处理数据，减弱其频谱泄露现象
        """
        img = np.log(img + 1)
        img = (img - np.mean(img)) / (np.std(img) + eps)
        return cos_window * img

    def randomWarp(self, img: np.ndarray) -> np.ndarray:
        """
        该函数对第一帧的目标框进行随机重定位，刚性形变，减轻漂移。
        """
        h, w = img.shape[:2]
        C = 0.1
        ang = np.random.uniform(-C, C)
        c, s = np.cos(ang), np.sin(ang)
        W = np.array(
            [
                [c + np.random.uniform(-C, C), -s + np.random.uniform(-C, C), 0],
                [s + np.random.uniform(-C, C), c + np.random.uniform(-C, C), 0],
            ]
        )
        center_warp = np.array([[w / 2], [h / 2]])
        tmp = np.sum(W[:, :2], axis=1).reshape((2, 1))
        W[:, 2:] = center_warp - center_warp * tmp
        warped = cv2.warpAffine(img, W, (w, h), cv2.BORDER_REFLECT)
        return warped
