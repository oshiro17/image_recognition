# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .diff import fused_diff_mask  # 循環しないように: diff は overlap を import しない

def overlap_mask(a_shape, transform, kind, b_shape):
    """B→A整列時のBの有効領域（重なり領域）マスクを作る。"""
    hA, wA = a_shape[:2]
    hB, wB = b_shape[:2]
    if transform is None:
        return np.ones((hA, wA), np.uint8)
    maskB = np.ones((hB, wB), np.uint8) * 255
    if kind == "AFFINE":
        warped = cv2.warpAffine(maskB, transform, (wA, hA),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    else:  # "H"
        warped = cv2.warpPerspective(maskB, transform, (wA, hA),
                                     flags=cv2.INTER_NEAREST)
    return (warped > 0).astype(np.uint8)

def masked_fused_diff(a, b, mask):
    """重なっていない領域は0に潰した差分マスク。"""
    th = fused_diff_mask(a, b)
    th[mask == 0] = 0
    return th