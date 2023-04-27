import os
import csv
import traceback

import cv2
import numpy as np

# from mosse_asddongmen import MOSSE

from mosse_yxj import MOSSE


# 返回排序后的所有帧的路径
def getSortedFrames(frames_dir) -> list[str]:
    frames = []
    for frame in sorted(os.listdir(frames_dir)):
        if os.path.splitext(frame)[1] == ".jpg":
            frames.append(os.path.join(frames_dir, frame))
    frames.sort()
    return frames


# 读取物体框  每一帧为顺序为 x y w h
def getGroundTruths(annotations_dir: str) -> list[list[int]]:
    groundtruth_path = os.path.join(annotations_dir, "groundtruth.txt")
    groundtruths: list[list[int]] = []
    with open(groundtruth_path) as f:
        for row in csv.reader(f):
            groundtruths.append([(int(float(i))) for i in row])
    return groundtruths


class PyTracker:
    def __init__(
        self, frames_dir: str, annotations_dir: str, tracker_type: str = "MOSSE"
    ):
        self.frames_dir = frames_dir
        self.tracker_type = tracker_type
        self.frames = getSortedFrames(frames_dir=frames_dir)
        self.groundtruths = getGroundTruths(annotations_dir=annotations_dir)

        if self.tracker_type == "MOSSE":
            self.tracker = MOSSE()
        else:
            raise NotImplementedError

    def tracking(self, video_path: str = None, verbose: bool = True) -> list[list[int]]:
        initial_frame = cv2.imread(self.frames[0])
        initial_groundtruth = self.groundtruths[0]
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

        # 首帧不用算
        for idx in range(1, len(self.frames)):
            current_frame = cv2.imread(self.frames[idx])
            bbox = self.tracker.update(current_frame, vis=verbose)
            poses.append(bbox)

            if verbose is True:
                x, y, w, h = bbox
                height, width = current_frame.shape[:2]
                if len(current_frame.shape) == 2:
                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
                score = self.tracker.score
                # apce = APCE(score)
                # psr = PSR(score)
                F_max = np.max(score)
                size = self.tracker.w, self.tracker.h
                score = cv2.resize(score, size)
                score -= score.min()
                score = score / score.max()
                score = (score * 255).astype(np.uint8)
                # score = 255 - score
                score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
                center = (int(x + w / 2), int(y + h / 2))
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
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    (255, 0, 0),
                    1,
                )

                cv2.imshow("demo", show_frame)
                if writer is not None:
                    writer.write(show_frame)
                cv2.waitKey(1)
        return poses


dataset_names = [
    "bicycle",
    "bolt",
    # 2 good 320x240 374/30 257,163,57,36
    # my: 32x64 [164,196)x[253,317)
    #     253.0,164.0,64.0,32.0
    # 32x32, 269.0,164.0,32.0,32.0
    "car",
    "cup",
    "david",
    "diving",
    "face",
    "gymnastics",
    "hand",
    "iceskater",
    "juice",
    # 11 good 640x360 228/30 328,189,53,50
    "jump",
    "singer",
    "sunshade",
    "torus",
    "woman",
]

if __name__ == "__main__":
    for dataset_name in ["car", "cup", "jump"]:
        annotations_path = f"data/{dataset_name}/"
        frames_path = f"data/{dataset_name}_frames/"
        video_path = f"output/{dataset_name}.mp4"

        # [开始跟踪算法]

        tracker = PyTracker(
            frames_dir=frames_path,
            annotations_dir=annotations_path,
            tracker_type="MOSSE",
        )
        try:
            poses = tracker.tracking(video_path=video_path)
            print(f"[debug] 跟踪 {len(poses)} 帧 (包括首帧{poses[0]})")
        except:
            print(traceback.format_exc())
