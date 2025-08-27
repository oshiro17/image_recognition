# -*- coding: utf-8 -*-
import cv2
import numpy as np

def _boxes_to_mask(boxes, shape, thickness=2):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, thickness)
    return mask

def cluster_dense_boxes(boxes, shape, dilate_iter=3, min_count=5, min_wh=25):
    if not boxes:
        return []
    h, w = shape[:2]
    mask = _boxes_to_mask(boxes, (h, w), thickness=2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dense = mask.copy()
    for _ in range(max(1, dilate_iter)):
        dense = cv2.dilate(dense, k, iterations=1)
    cnts = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    clusters = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_wh or bh < min_wh:
            continue
        cx1, cy1, cx2, cy2 = x, y, x + bw, y + bh
        cnt = 0
        for (bx, by, bw2, bh2) in boxes:
            cx = bx + bw2 // 2
            cy = by + bh2 // 2
            if (cx1 <= cx <= cx2) and (cy1 <= cy <= cy2):
                cnt += 1
        if cnt >= min_count:
            clusters.append((x, y, bw, bh, cnt))
    return clusters