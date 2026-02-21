from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple

import cv2


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


def open_video(path: str) -> Tuple[cv2.VideoCapture, VideoMeta]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video. Unsupported codec or bad file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, VideoMeta(width=width, height=height, fps=fps, frame_count=frame_count)


def read_frame_at(cap: cv2.VideoCapture, index: int) -> Tuple[bool, "cv2.Mat"]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    return cap.read()


def iter_frames(cap: cv2.VideoCapture, start: int = 0) -> Generator[Tuple[int, "cv2.Mat"], None, None]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1


def make_video_writer(path: str, meta: VideoMeta) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, meta.fps, (meta.width, meta.height))
