# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .cluster import cluster_dense_boxes

def make_boxes_overlay_transparent(imgA, imgB, boxes, alpha=0.5, draw_border=True):
    out = imgA.copy()
    for (x, y, w, h) in boxes:
        x = max(0, int(x)); y = max(0, int(y))
        w = max(1, int(w)); h = max(1, int(h))
        x2 = min(out.shape[1], x + w)
        y2 = min(out.shape[0], y + h)
        if x2 <= x or y2 <= y:
            continue
        roiA = out[y:y2, x:x2]
        roiB = imgB[y:y2, x:x2]
        if roiA.shape[:2] != roiB.shape[:2]:
            roiB = cv2.resize(roiB, (roiA.shape[1], roiA.shape[0]), interpolation=cv2.INTER_AREA)
        blended = cv2.addWeighted(roiA, 1.0 - alpha, roiB, alpha, 0.0)
        out[y:y2, x:x2] = blended
        if draw_border:
            cv2.rectangle(out, (x, y), (x2, y2), (0, 0, 255), 2)
    return out

def draw_clusters_only(base_img, clusters, color=(0, 0, 255), thickness=6, show_count=True, blend_with=None, alpha=0.35):
    vis = base_img.copy()
    if blend_with is not None:
        b = cv2.resize(blend_with, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_AREA)
        vis = cv2.addWeighted(vis, 1.0 - alpha, b, alpha, 0.0)
    for (x, y, w, h, cnt) in clusters:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        if show_count:
            cv2.putText(vis, f"x{cnt}", (x + 6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis

def make_dense_overlay(base_img, overlay_img, boxes, alpha=0.5,
                       dilate_iter=3, min_count=5, min_wh=25,
                       color=(0, 0, 255), thickness=6, show_count=True):
    vis = make_boxes_overlay_transparent(base_img, overlay_img, boxes, alpha=alpha, draw_border=True)
    clusters = cluster_dense_boxes(boxes, base_img.shape, dilate_iter=dilate_iter, min_count=min_count, min_wh=min_wh)
    for (x, y, w, h, cnt) in clusters:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        if show_count:
            cv2.putText(vis, f"x{cnt}", (x + 6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis, clusters