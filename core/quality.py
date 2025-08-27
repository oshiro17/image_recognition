# -*- coding: utf-8 -*-
import cv2
import numpy as np

def quality_harmonize_pair(a: np.ndarray, b: np.ndarray,
                           target_short=720,
                           match_sharp=True,
                           match_hist=True,
                           denoise='none'):
    # サイズ合わせ
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    # 共通ダウンサンプル
    h, w = a.shape[:2]
    short = min(h, w)
    if target_short is not None and short > target_short:
        r = target_short / float(short)
        new_size = (int(w * r), int(h * r))
        a = cv2.resize(a, new_size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, new_size, interpolation=cv2.INTER_AREA)

    # シャープネス整合
    if match_sharp:
        def varlap(x):
            return cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        va, vb = varlap(a), varlap(b)

        def blur_light(img, sigma=1.0):
            sigma = max(0.5, min(3.0, float(sigma)))
            ksize = int(2 * round(3 * sigma) + 1)
            return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

        if va > vb * 1.10:
            a = blur_light(a, 1.0)
        elif vb > va * 1.10:
            b = blur_light(b, 1.0)

    # 明るさ/コントラスト整合
    if match_hist:
        def match_meanstd(src, ref):
            gs = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gr = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
            ms, ss = gs.mean(), gs.std() + 1e-6
            mr, sr = gr.mean(), gr.std() + 1e-6
            gain, bias = (sr / ss), (mr - ms * (sr / ss))
            out = np.clip(gs * gain + bias, 0, 255).astype(np.uint8)
            return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        a = match_meanstd(a, b)
        b = match_meanstd(b, a)

    # ノイズ調整
    if denoise == 'light':
        a = cv2.fastNlMeansDenoisingColored(a, None, 3, 3, 7, 15)
        b = cv2.fastNlMeansDenoisingColored(b, None, 3, 3, 7, 15)
    elif denoise == 'add_noise':
        rng = np.random.default_rng(0)
        def add_gnoise(img, sigma=2.0):
            n = rng.normal(0, sigma, img.shape).astype(np.float32)
            return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        a, b = add_gnoise(a), add_gnoise(b)

    return a, b