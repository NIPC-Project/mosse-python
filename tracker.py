import os
import csv

import cv2
import numpy as np

from my_mosse import MOSSE


def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(
            region[2:4] - region[4:6]
        )
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2  # 中心坐标
        cy = y + h / 2

    return cx - w / 2, cy - h / 2, w, h


# 返回排序后的所有帧的路径
def getSortedFrames(frames_dir) -> list[str]:
    frames = []
    for frame in sorted(os.listdir(frames_dir)):
        if os.path.splitext(frame)[1] == ".jpg":
            frames.append(os.path.join(frames_dir, frame))
    frames.sort()
    return frames


# 读取物体框  每一帧为顺序为 x y w h
def getGroundTruths(frames_dir: str) -> list[list[int]]:
    groundtruth_path = os.path.join(frames_dir, "groundtruth.txt")
    groundtruths: list[list[int]] = []
    with open(groundtruth_path) as f:
        for row in csv.reader(f):
            groundtruths.append([(int(float(i))) for i in row])
    return groundtruths


class PyTracker:
    def __init__(self, frames_dir: str, tracker_type: str = "MOSSE"):
        self.frames_dir = frames_dir
        self.tracker_type = tracker_type
        self.frames = getSortedFrames(frames_dir=frames_dir)
        self.groundtruths = getGroundTruths(frames_dir=frames_dir)

        if self.tracker_type == "MOSSE":
            self.tracker = MOSSE()
        else:
            raise NotImplementedError

    def tracking(self, video_path: str = None, verbose: bool = True) -> list[list[int]]:

        initial_frame = cv2.imread(self.frames[0])
        initial_groundtruth = self.groundtruths[0]
        x1, y1, w, h = initial_groundtruth
        poses = [initial_groundtruth]
        self.tracker.init(first_frame=initial_frame, bbox=initial_groundtruth)

        writer = None
        if verbose is True and video_path is not None:
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc("H", "2", "6", "4"),
                30,
                (initial_frame.shape[1], initial_frame.shape[0]),
            )

        for idx in range(len(self.frames)):
            if idx != 0:
                current_frame = cv2.imread(self.frames[idx])
                height, width = current_frame.shape[:2]
                bbox = self.tracker.update(current_frame, vis=verbose)
                x1, y1, w, h = bbox
                if verbose is True:
                    if len(current_frame.shape) == 2:
                        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
                    score = self.tracker.score
                    # apce = APCE(score)
                    # psr = PSR(score)
                    F_max = np.max(score)
                    size = self.tracker.crop_size
                    score = cv2.resize(score, size)
                    score -= score.min()
                    score = score / score.max()
                    score = (score * 255).astype(np.uint8)
                    # score = 255 - score
                    score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
                    center = (int(x1 + w / 2), int(y1 + h / 2))
                    x0, y0 = center
                    x0 = np.clip(x0, 0, width - 1)
                    y0 = np.clip(y0, 0, height - 1)
                    center = (x0, y0)
                    xmin = int(center[0]) - size[0] // 2
                    xmax = int(center[0]) + size[0] // 2 + size[0] % 2
                    ymin = int(center[1]) - size[1] // 2
                    ymax = int(center[1]) + size[1] // 2 + size[1] % 2
                    left = abs(xmin) if xmin < 0 else 0
                    xmin = 0 if xmin < 0 else xmin
                    right = width - xmax
                    xmax = width if right < 0 else xmax
                    right = size[0] + right if right < 0 else size[0]
                    top = abs(ymin) if ymin < 0 else 0
                    ymin = 0 if ymin < 0 else ymin
                    down = height - ymax
                    ymax = height if down < 0 else ymax
                    down = size[1] + down if down < 0 else size[1]
                    score = score[top:down, left:right]
                    crop_img = current_frame[ymin:ymax, xmin:xmax]
                    score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
                    current_frame[ymin:ymax, xmin:xmax] = score_map
                    show_frame = cv2.rectangle(
                        current_frame,
                        (int(x1), int(y1)),
                        (int(x1 + w), int(y1 + h)),
                        (255, 0, 0),
                        1,
                    )

                    cv2.imshow("demo", show_frame)
                    if writer is not None:
                        writer.write(show_frame)
                    cv2.waitKey(1)

            poses.append(np.array([int(x1), int(y1), int(w), int(h)]))
        return poses


if __name__ == "__main__":
    tracker = PyTracker(frames_dir="./input/bicycle/", tracker_type="MOSSE")
    poses = tracker.tracking(video_path="./output/bicycle.mp4")
    print(f"[debug] {poses[2]}")
