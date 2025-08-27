# core/diff.py
# 差分抽出まわり（absdiff→二値→輪郭＝difference_boxes、照明に強いfused_diff_mask、マスクから矩形抽出）

import cv2
import numpy as np

def grad_mag(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def census8(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    out = np.zeros((h, w), dtype=np.uint8)
    bit = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
            cmp = (g > shifted).astype(np.uint8)
            out = cv2.bitwise_or(out, ((cmp << bit) & 0xFF))
            bit = (bit + 1) % 8
    return out

def fused_diff_mask(a, b):
    """勾配+センサスの差分を融合→適応二値化→モルフォ処理。
       照明差に比較的強い差分マスクを返す（uint8, 0/255）。"""
    d1 = cv2.absdiff(grad_mag(a), grad_mag(b))
    d2 = cv2.absdiff(census8(a), census8(b))
    fused = cv2.addWeighted(d1, 0.6, d2, 0.4, 0)
    h, w = fused.shape
    ks = max(15, (min(h, w)//40)|1)  # 奇数カーネル
    th = cv2.adaptiveThreshold(fused, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, ks, -3)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    return th

def difference_boxes(imgA, imgB, min_wh=15, bin_thresh=None):
    """厳密整列が前提のときの基本差分。
       absdiff→(大津 or 固定閾値)→開→膨張→輪郭→外接矩形。"""
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gA, gB)

    if bin_thresh is None:
        _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(diff, bin_thresh, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.dilate(th, k, iterations=1)

    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    vis = imgA.copy()
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_wh and h < min_wh:
            continue
        boxes.append((x, y, w, h))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return vis, th, boxes

def boxes_from_mask(imgA, mask, min_wh=15):
    """二値マスクから輪郭→外接矩形→可視化。"""
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    vis = imgA.copy()
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_wh and h < min_wh:
            continue
        boxes.append((x, y, w, h))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return vis, boxes

__all__ = [
    "difference_boxes",
    "fused_diff_mask",
    "boxes_from_mask",
    "grad_mag",
    "census8",
]