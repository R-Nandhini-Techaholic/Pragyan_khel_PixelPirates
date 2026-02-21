from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class TrackSelection:
    track_id: int
    bbox: Tuple[int, int, int, int]


def _boxes_from_result(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], [], []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    return xyxy, conf, ids


def get_candidate_boxes(result, max_candidates: int = 5) -> List[Tuple[Tuple[int, int, int, int], Optional[int], Optional[float]]]:
    xyxy, conf, ids = _boxes_from_result(result)
    if len(xyxy) == 0:
        return []

    indices = list(range(len(xyxy)))
    if conf is not None:
        indices.sort(key=lambda i: float(conf[i]), reverse=True)
    if max_candidates is not None:
        indices = indices[:max_candidates]

    candidates = []
    for i in indices:
        x1, y1, x2, y2 = xyxy[i]
        bbox = (int(x1), int(y1), int(x2), int(y2))
        track_id = int(ids[i]) if ids is not None else None
        score = float(conf[i]) if conf is not None else None
        candidates.append((bbox, track_id, score))
    return candidates


def choose_target_from_click(result, click_x: int, click_y: int) -> Optional[TrackSelection]:
    xyxy, _, ids = _boxes_from_result(result)
    if len(xyxy) == 0:
        return None

    best_idx = None
    best_dist = None

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box
        if click_x < x1 or click_x > x2 or click_y < y1 or click_y > y2:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = (click_x - cx) ** 2 + (click_y - cy) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is None:
        return None

    if ids is None:
        return None

    x1, y1, x2, y2 = xyxy[best_idx]
    return TrackSelection(track_id=int(ids[best_idx]), bbox=(int(x1), int(y1), int(x2), int(y2)))


def find_bbox_for_track(result, track_id: int) -> Optional[Tuple[int, int, int, int]]:
    xyxy, _, ids = _boxes_from_result(result)
    if ids is None:
        return None

    for i, t_id in enumerate(ids):
        if int(t_id) == int(track_id):
            x1, y1, x2, y2 = xyxy[i]
            return int(x1), int(y1), int(x2), int(y2)

    return None


def blur_except_bbox(frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    if bbox is None:
        return blurred

    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return blurred

    blurred[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return blurred


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.fastNlMeansDenoisingColored(enhanced, None, 7, 7, 7, 21)


def _grabcut_mask(frame: np.ndarray, bbox: Tuple[int, int, int, int], iterations: int = 1) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None

    rect = (x1, y1, x2 - x1, y2 - y1)
    mask = np.zeros(frame.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(frame, mask, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    return mask2


def apply_focus_effect(
    frame: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    use_grabcut: bool = False,
) -> np.ndarray:
    if bbox is None:
        return cv2.GaussianBlur(frame, (35, 35), 0)

    if not use_grabcut:
        return blur_except_bbox(frame, bbox)

    mask = _grabcut_mask(frame, bbox, iterations=1)
    if mask is None:
        return blur_except_bbox(frame, bbox)

    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    output = blurred.copy()
    output[mask == 1] = frame[mask == 1]
    return output


def draw_boxes(frame: np.ndarray, result) -> np.ndarray:
    xyxy, conf, ids = _boxes_from_result(result)
    output = frame.copy()
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box.astype(int)
        label = ""
        if ids is not None:
            label = f"ID {int(ids[i])}"
        if conf is not None:
            label = f"{label} {conf[i]:.2f}".strip()
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label:
            cv2.putText(output, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output


def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    return YOLO(model_name)


def find_bbox_by_proximity(
    result,
    reference_bbox: Optional[Tuple[int, int, int, int]],
    max_distance: Optional[float] = None,
) -> Optional[Tuple[int, int, int, int]]:
    if reference_bbox is None:
        return None

    xyxy, _, _ = _boxes_from_result(result)
    if len(xyxy) == 0:
        return None

    rx1, ry1, rx2, ry2 = reference_bbox
    ref_cx = (rx1 + rx2) / 2.0
    ref_cy = (ry1 + ry2) / 2.0

    best_idx = None
    best_dist = None
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = ((ref_cx - cx) ** 2 + (ref_cy - cy) ** 2) ** 0.5
        if max_distance is not None and dist > max_distance:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is None:
        return None

    x1, y1, x2, y2 = xyxy[best_idx]
    return int(x1), int(y1), int(x2), int(y2)


def find_bbox_and_id_by_proximity(
    result,
    reference_bbox: Optional[Tuple[int, int, int, int]],
    max_distance: Optional[float] = None,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[int]]:
    if reference_bbox is None:
        return None, None

    xyxy, _, ids = _boxes_from_result(result)
    if len(xyxy) == 0:
        return None, None

    rx1, ry1, rx2, ry2 = reference_bbox
    ref_cx = (rx1 + rx2) / 2.0
    ref_cy = (ry1 + ry2) / 2.0

    best_idx = None
    best_dist = None
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = ((ref_cx - cx) ** 2 + (ref_cy - cy) ** 2) ** 0.5
        if max_distance is not None and dist > max_distance:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is None:
        return None, None

    x1, y1, x2, y2 = xyxy[best_idx]
    new_id = int(ids[best_idx]) if ids is not None else None
    return (int(x1), int(y1), int(x2), int(y2)), new_id


class AppearanceMatcher:
    def __init__(self, device: str = "cpu", use_pretrained: bool = True) -> None:
        self.mode = "hist"
        self.device = device
        self.backbone = None
        self.mean = None
        self.std = None

        if not use_pretrained:
            return

        try:
            import torch
            from torchvision import models
            from torchvision.models import MobileNet_V3_Small_Weights

            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
            self.backbone = torch.nn.Sequential(
                model.features,
                torch.nn.AdaptiveAvgPool2d(1),
            ).to(device)
            self.backbone.eval()
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
            self.mode = "torch"
        except Exception:
            self.mode = "hist"

    def embed_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if self.mode == "hist":
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist.astype(np.float32)

        import torch

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(crop).to(self.device).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            feats = self.backbone(tensor).squeeze()
        if feats.ndim == 0:
            return None
        vec = feats.flatten()
        vec = vec / (vec.norm() + 1e-6)
        return vec.cpu().numpy()

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return -1.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def best_match(
        self,
        frame: np.ndarray,
        candidates: List[Tuple[Tuple[int, int, int, int], Optional[int], Optional[float]]],
        target_embedding: np.ndarray,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[int], float]:
        best_bbox = None
        best_id = None
        best_sim = -1.0
        for bbox, track_id, _ in candidates:
            emb = self.embed_crop(frame, bbox)
            if emb is None:
                continue
            sim = float(np.dot(emb, target_embedding))
            if sim > best_sim:
                best_sim = sim
                best_bbox = bbox
                best_id = track_id
        return best_bbox, best_id, best_sim
