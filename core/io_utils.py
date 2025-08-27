# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"画像が読めません: {path}")
    return img

def resize_to(img: np.ndarray, size_hw) -> np.ndarray:
    return cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)

def ensure_same_size(a: np.ndarray, b: np.ndarray):
    if a.shape[:2] != b.shape[:2]:
        b = resize_to(b, a.shape[:2])
    return a, b

def save_side_by_side(imgA: np.ndarray, imgB: np.ndarray, out_path: str) -> None:
    h = max(imgA.shape[0], imgB.shape[0])
    w = imgA.shape[1] + imgB.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:imgA.shape[0], :imgA.shape[1]] = imgA
    canvas[:imgB.shape[0], imgA.shape[1]:imgA.shape[1]+imgB.shape[1]] = imgB
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas)

def save_alpha_overlay(imgA: np.ndarray, imgB: np.ndarray, out_path: str, alpha: float = 0.35) -> None:
    b = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]), interpolation=cv2.INTER_AREA)
    blended = cv2.addWeighted(imgA, 1.0 - alpha, b, alpha, 0.0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, blended)