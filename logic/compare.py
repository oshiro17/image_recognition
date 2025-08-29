# logic/compare.py
from typing import Dict, Any, Optional, List
import numpy as np
import cv2

from core.io_utils import ensure_same_size
from core.align import camera_misaligned, align_ecc, align_homography
from core.vis import make_boxes_overlay_transparent, draw_clusters_only
from core.cluster import cluster_dense_boxes
from core.diff import difference_boxes, fused_diff_mask, boxes_from_mask
from core.quality import quality_harmonize_pair
from core.yolo import detect_objects, draw_detections
from logic.yolo_changes import analyze_yolo_changes

def compare_to_baseline(baseline_bgr: np.ndarray, current_bgr: np.ndarray, cfg: Dict[str,Any], baseline_yolo: List[Dict[str,Any]]):
    # 画質整合
    a, b = quality_harmonize_pair(baseline_bgr, current_bgr,
                                  target_short=cfg.get("target_short", 720),
                                  match_sharp=True, match_hist=True, denoise='none')
    a, b = ensure_same_size(a, b)

    # ズレ判定→整列
    aligned_method = "NONE"
    if cfg.get("align_mode","AUTO") != "OFF":
        misaligned, _d = camera_misaligned(a, b, shift_px_thresh=cfg.get("shift_px_thresh", 6.0), homography_checks=True)
        if misaligned:
            if cfg.get("align_mode","AUTO") in ("AUTO","ECC"):
                b_ecc, _warp = align_ecc(a, b)
                if b_ecc is not None:
                    b = b_ecc; aligned_method = "ECC"
            if aligned_method == "NONE" and cfg.get("align_mode","AUTO") in ("AUTO","H"):
                b_h, _H = align_homography(a, b)
                if b_h is not None:
                    b = b_h; aligned_method = "H"

    # 差分ボックス
    if aligned_method == "NONE":
        vis_boxes, th_mask, boxes = difference_boxes(a, b, min_wh=cfg.get("min_wh", 15), bin_thresh=None)
    else:
        th_mask = fused_diff_mask(a, b)
        vis_boxes, boxes = boxes_from_mask(a, th_mask, min_wh=cfg.get("min_wh", 15))

    diff_ratio = float((th_mask>0).sum()) / float(th_mask.size)
    clusters = cluster_dense_boxes(boxes, a.shape, dilate_iter=3, min_count=6, min_wh=30)
    clusters_vis = draw_clusters_only(a, clusters, color=(0,0,255), thickness=6, show_count=True)

    # SSIM
    try:
        from skimage.metrics import structural_similarity as ssim
        ss_val = ssim(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    except Exception:
        ss_val = None

    # 色ターゲット監視（cfg["targets"] は UI 側で生成）
    target_results = []
    for t in (cfg.get("targets", []) or []):
        try:
            curr_ratio = _ratio_for_target(b, t)
            base_ratio = float(t.get("base_ratio", 0.0))
            direction = t.get("direction", "decrease")
            thr = float(t.get("threshold_pct", 5.0)) / 100.0
            delta = curr_ratio - base_ratio
            if direction == "increase":
                alert = (delta >= thr); vanished = False
            else:
                alert = (-delta >= thr); vanished = (curr_ratio <= 0.001)
            target_results.append({
                "name": t.get("name", "target"),
                "curr_ratio": curr_ratio,
                "base_ratio": base_ratio,
                "delta": delta,
                "direction": direction,
                "threshold_pct": t.get("threshold_pct", 5.0),
                "alert": bool(alert),
                "vanished": bool(vanished)
            })
        except Exception as e:
            target_results.append({"name": t.get("name","target"), "error": str(e)})

    # YOLO（現在）
    yolo_conf = float(cfg.get("yolo_conf", 0.5))
    det_cur = detect_objects(b, conf=yolo_conf)
    yolo_vis = draw_detections(b, det_cur)
    yolo_changes = analyze_yolo_changes(baseline_yolo or [], det_cur, iou_thr=0.3)

    return {
        "A": a, "B": b,
        "vis_boxes": vis_boxes, "mask": th_mask, "clusters_vis": clusters_vis,
        "boxes": boxes, "diff_ratio": diff_ratio, "ssim": ss_val,
        "aligned": aligned_method,
        "target_results": target_results,
        "yolo_detections": det_cur, "yolo_vis": yolo_vis, "yolo_changes": yolo_changes
    }

# ===== 色ターゲット用（UIの定義と一致させる簡易版） =====
def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

def _target_mask_for_img(img: np.ndarray, target: dict, roi=None) -> np.ndarray:
    lab = _lab(img)
    h, w = lab.shape[:2]
    if roi is not None:
        x, y, rw, rh = [int(v) for v in roi]
        x = max(0, x); y = max(0, y); rw = max(1, rw); rh = max(1, rh)
        x2 = min(w, x + rw); y2 = min(h, y + rh)
        sub = lab[y:y2, x:x2]; off = (x, y)
    else:
        sub = lab; off = (0, 0)
    meanL, meana, meanb = target.get("mean", (0, 0, 0))
    stdL, stda, stdb = target.get("std", (1, 1, 1))
    k = float(target.get("k_sigma", 2.5))
    use_l = bool(target.get("use_l", False))
    L = sub[..., 0]; A = sub[..., 1]; B = sub[..., 2]
    da = (A - meana) / (stda + 1e-6)
    db = (B - meanb) / (stdb + 1e-6)
    dist2 = da * da + db * db
    if use_l:
        dL = (L - meanL) / (stdL + 1e-6)
        dist2 = dist2 + dL * dL
    mask = (dist2 <= (k * k)).astype(np.uint8) * 255
    out = np.zeros((h, w), np.uint8)
    out[off[1]:off[1] + mask.shape[0], off[0]:off[0] + mask.shape[1]] = mask
    return out

def _ratio_for_target(img: np.ndarray, target: dict) -> float:
    roi = target.get("roi")
    m = _target_mask_for_img(img, target, roi)
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        denom = max(1, w * h)
        return float((m[y:y+h, x:x+w] > 0).sum()) / float(denom)
    else:
        return float((m > 0).sum()) / float(m.size)