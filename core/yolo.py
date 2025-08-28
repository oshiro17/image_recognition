# core/yolo.py
# -*- coding: utf-8 -*-
"""
YOLOv8 を使った物体検出モジュール（改良版）
- 物体検出（柔軟なパラメータ）
- 差分マスクとの重なりチェック
- 基準画像の物体一覧スナップショットと現在の比較
"""

from __future__ import annotations
import os
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # pip install ultralytics
except Exception:  # ImportError を含む広めの捕捉（環境差対策）
    YOLO = None

__all__ = [
    "load_model",
    "detect_objects",
    "draw_detections",
    "check_overlap_with_diff",
    "detect_and_check",
    "snapshot_baseline_objects",
    "compare_objects_vs_baseline",
]

# ----------------- 初期化（キャッシュ） -----------------
_model_cache: Optional[Any] = None
_model_name_cache: Optional[str] = None


def _ensure_ultralytics():
    if YOLO is None:
        raise ImportError(
            "ultralytics が見つかりません。`pip install ultralytics` を実行してください。"
        )


def load_model(model_name: str = "yolov8n.pt") -> Any:
    """
    YOLOモデルをロード（キャッシュ付き）
    model_name は環境変数 ULTRA_YOLO_MODEL があればそれを優先
    """
    global _model_cache, _model_name_cache
    _ensure_ultralytics()
    wanted = os.environ.get("ULTRA_YOLO_MODEL", model_name)
    if _model_cache is None or _model_name_cache != wanted:
        _model_cache = YOLO(wanted)
        _model_name_cache = wanted
    return _model_cache


# ----------------- 基本の物体検出 -----------------
def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def detect_objects(
    img_bgr: np.ndarray,
    conf: float = 0.5,
    iou: float = 0.5,
    imgsz: int | Tuple[int, int] = 640,
    classes: Optional[List[int]] = None,
    device: Optional[str] = None,
    max_det: int = 300,
) -> List[Dict[str, Any]]:
    """
    画像から物体検出を行う（BGR入力）
    戻り値: list of {id, name, conf, box=(x1,y1,x2,y2), area}
    """
    model = load_model()
    h, w = img_bgr.shape[:2]

    # ultralytics は RGB 前提
    rgb = img_bgr[..., ::-1]

    results = model.predict(
        rgb,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=classes,
        device=device,
        max_det=max_det,
        verbose=False,
    )
    objects: List[Dict[str, Any]] = []
    # 複数バッチにも備える
    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        names = getattr(model, "names", {})
        for b in r.boxes:
            cls_id = int(b.cls[0])
            conf_val = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, w, h)
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            objects.append(
                {
                    "id": cls_id,
                    "name": names.get(cls_id, str(cls_id)),
                    "conf": conf_val,
                    "box": (x1, y1, x2, y2),
                    "area": int(area),
                }
            )
    return objects


# ----------------- 可視化 -----------------
def draw_detections(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    show_score: bool = True,
    color: Tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    検出結果を矩形で描画
    color=None の場合はクラス毎に自動色
    """
    vis = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name, conf = det["name"], det["conf"]
        if color is None:
            # クラスIDベースの簡易色
            cid = int(det.get("id", 0))
            col = ((37 * cid) % 256, (17 * cid) % 256, (97 * cid) % 256)
        else:
            col = color
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thickness)
        label = f"{name} {conf:.2f}" if show_score else name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1t = max(0, y1 - th - 3)
        cv2.rectangle(vis, (x1, y1t), (x1 + tw + 4, y1t + th + 4), col, -1)
        cv2.putText(vis, label, (x1 + 2, y1t + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


# ----------------- 差分と物体の重なりチェック -----------------
def check_overlap_with_diff(
    detections: List[Dict[str, Any]],
    diff_mask: np.ndarray,
    min_overlap: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    差分マスクと検出された物体の領域が重なっているかを判定
    min_overlap: 物体領域の何割以上が差分に含まれていたら「異常」とする
    戻り値: [{"name","id","box","overlap"} ...]  異常が見つかった物体（重なり率つき）
    """
    alerts: List[Dict[str, Any]] = []
    if diff_mask is None or diff_mask.size == 0:
        return alerts

    h, w = diff_mask.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, w, h)
        roi_mask = diff_mask[y1:y2, x1:x2]
        if roi_mask.size == 0:
            continue
        overlap_ratio = float((roi_mask > 0).sum()) / float(roi_mask.size)
        if overlap_ratio >= min_overlap:
            alerts.append(
                {
                    "id": det["id"],
                    "name": det["name"],
                    "box": (x1, y1, x2, y2),
                    "overlap": overlap_ratio,
                }
            )
    return alerts


# ----------------- ヘルパまとめ -----------------
def detect_and_check(
    img_bgr: np.ndarray,
    diff_mask: np.ndarray,
    conf: float = 0.5,
    iou: float = 0.5,
    imgsz: int | Tuple[int, int] = 640,
    classes: Optional[List[int]] = None,
    device: Optional[str] = None,
    max_det: int = 300,
    min_overlap: float = 0.1,
) -> Dict[str, Any]:
    """
    1枚の画像について:
      - YOLOで物体を検出
      - 差分マスクと重なりがある物体をチェック
    戻り値:
      {
        "detections": [...],
        "alerts": [{"name","id","box","overlap"}, ...]
      }
    """
    detections = detect_objects(
        img_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=classes,
        device=device,
        max_det=max_det,
    )
    alerts = check_overlap_with_diff(detections, diff_mask, min_overlap=min_overlap)
    return {"detections": detections, "alerts": alerts}


# ----------------- 基準スナップショットと比較 -----------------
def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = max(1e-6, area_a + area_b - inter)
    return float(inter) / float(union)


def snapshot_baseline_objects(
    img_bgr: np.ndarray,
    conf: float = 0.5,
    iou: float = 0.5,
    imgsz: int | Tuple[int, int] = 640,
    classes: Optional[List[int]] = None,
    device: Optional[str] = None,
    max_det: int = 300,
) -> List[Dict[str, Any]]:
    """
    基準画像の物体一覧を保存しておくための検出ヘルパ
    """
    return detect_objects(
        img_bgr, conf=conf, iou=iou, imgsz=imgsz, classes=classes, device=device, max_det=max_det
    )


def compare_objects_vs_baseline(
    baseline_dets: List[Dict[str, Any]],
    current_dets: List[Dict[str, Any]],
    iou_match: float = 0.5,
) -> Dict[str, Any]:
    """
    基準と現在の検出結果を突き合わせ：
      - 新規出現（基準に無いクラスが現れた）
      - 消失（基準にあって現在に無いクラス or 大きく位置がズレた＝別物と判断）
    戻り:
      {
        "appeared": [{"name","id","box"}...],
        "disappeared": [{"name","id","box"}...]
      }
    """
    appeared: List[Dict[str, Any]] = []
    disappeared: List[Dict[str, Any]] = []

    # クラス毎にグルーピング
    from collections import defaultdict
    base_by_cls = defaultdict(list)
    cur_by_cls = defaultdict(list)
    for d in baseline_dets:
        base_by_cls[d["id"]].append(d)
    for d in current_dets:
        cur_by_cls[d["id"]].append(d)

    # 消失判定：基準にある box が、現在どの box とも IoU>=閾値 でマッチしない
    for cid, base_items in base_by_cls.items():
        cur_items = cur_by_cls.get(cid, [])
        for b in base_items:
            matched = False
            for c in cur_items:
                if _iou_xyxy(b["box"], c["box"]) >= iou_match:
                    matched = True
                    break
            if not matched:
                disappeared.append({"id": b["id"], "name": b["name"], "box": b["box"]})

    # 出現判定：現在の box が、基準どの box ともマッチしない
    for cid, cur_items in cur_by_cls.items():
        base_items = base_by_cls.get(cid, [])
        for c in cur_items:
            matched = False
            for b in base_items:
                if _iou_xyxy(b["box"], c["box"]) >= iou_match:
                    matched = True
                    break
            if not matched:
                appeared.append({"id": c["id"], "name": c["name"], "box": c["box"]})

    return {"appeared": appeared, "disappeared": disappeared}