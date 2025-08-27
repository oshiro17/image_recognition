# -*- coding: utf-8 -*-
import math
import cv2
import numpy as np

def phase_correlation_shift(g1: np.ndarray, g2: np.ndarray):
    g1f = g1.astype(np.float32)
    g2f = g2.astype(np.float32)
    win = cv2.createHanningWindow((g1.shape[1], g1.shape[0]), cv2.CV_32F)
    s = cv2.phaseCorrelate(g1f * win, g2f * win)[0]  # (dx, dy)
    return s

def estimate_homography(a_gray: np.ndarray, b_gray: np.ndarray):
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(a_gray, None)
    k2, d2 = orb.detectAndCompute(b_gray, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    m = bf.knnMatch(d1, d2, k=2)
    good = [mm[0] for mm in m if len(mm) == 2 and mm[0].distance < 0.75*mm[1].distance]
    if len(good) < 12:
        return None
    src_pts = np.float32([k1[g.queryIdx].pt for g in good]).reshape(-1,1,2)
    dst_pts = np.float32([k2[g.trainIdx].pt for g in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H

def is_near_identity_homography(H, tol_trans=6.0, tol_rot_deg=2.0, tol_scale=0.05, tol_persp=0.002) -> bool:
    if H is None:
        return False
    H = H / H[2,2]
    tx, ty = H[0,2], H[1,2]
    if abs(tx) > tol_trans or abs(ty) > tol_trans:
        return False
    a,b,c,d = H[0,0], H[0,1], H[1,0], H[1,1]
    rot_rad = math.atan2(c, a)
    rot_deg = abs(rot_rad * 180.0 / math.pi)
    if rot_deg > tol_rot_deg:
        return False
    sx = math.sqrt(a*a + c*c)
    sy = math.sqrt(b*b + d*d)
    if not (1.0 - tol_scale <= sx <= 1.0 + tol_scale): return False
    if not (1.0 - tol_scale <= sy <= 1.0 + tol_scale): return False
    if abs(H[2,0]) > tol_persp or abs(H[2,1]) > tol_persp:
        return False
    return True

def camera_misaligned(imgA: np.ndarray, imgB: np.ndarray, shift_px_thresh=6.0, homography_checks=True):
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    dx, dy = phase_correlation_shift(gA, gB)
    shift_norm = float((dx**2 + dy**2) ** 0.5)
    if shift_norm > shift_px_thresh:
        return True, {"phase_shift": (dx, dy), "shift_norm": shift_norm, "H_ok": None}
    if homography_checks:
        H = estimate_homography(gA, gB)
        H_ok = is_near_identity_homography(H)
        if not H_ok:
            return True, {"phase_shift": (dx, dy), "shift_norm": shift_norm, "H_ok": False}
        return False, {"phase_shift": (dx, dy), "shift_norm": shift_norm, "H_ok": True}
    return False, {"phase_shift": (dx, dy), "shift_norm": shift_norm, "H_ok": None}

def align_ecc(a: np.ndarray, b: np.ndarray):
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2,3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
        _, warp = cv2.findTransformECC(ag, bg, warp, cv2.MOTION_AFFINE, criteria)
        aligned = cv2.warpAffine(b, warp, (a.shape[1], a.shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        return aligned, warp
    except cv2.error:
        return None, None

def align_homography(a: np.ndarray, b: np.ndarray):
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(ag, None)
    k2, d2 = orb.detectAndCompute(bg, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None, None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p for p,q in m if p.distance < 0.75*q.distance]
    if len(good) < 12:
        return None, None
    src = np.float32([k1[p.queryIdx].pt for p in good]).reshape(-1,1,2)
    dst = np.float32([k2[p.trainIdx].pt for p in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    if H is None:
        return None, None
    aligned = cv2.warpPerspective(b, H, (a.shape[1], a.shape[0]))
    return aligned, H