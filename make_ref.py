# make_ref.py
# -*- coding: utf-8 -*-
"""
カメラ位置ズレ検出 → ズレがない場合だけ差分可視化（赤枠）を実行するスクリプト。

============================
■ なにをするスクリプトか
============================
2枚の画像（A, B）が与えられたとき、まず「カメラ位置が変わっていないか」を判定します。
- 位置のズレが **ある** と判断した場合：
  → 処理を打ち切り、「カメラの位置が違う画像です」とメッセージを出して終了します。
- 位置のズレが **ない（＝ほぼ同じ構図）** と判断した場合：
  → 2枚の差分を抽出し、**差分領域を赤い枠で囲った画像**（diff_boxes.png）と、
     **差分の二値マスク画像**（diff_mask.png）を `results/` に出力します。

============================
■ 想定する入力
============================
- A: 画像1（例: test1a.png）
- B: 画像2（例: test1b.png）
サイズが異なる場合は A に合わせて B をリサイズします。

============================
■ 出力ファイル（results/ 配下）
============================
- diff_boxes.png : 画像A上に、差分領域を赤枠で可視化したもの（人に見せる用）。
- diff_mask.png  : 白=差分あり / 黒=差分なし の二値マスク（後段処理や判定に使う用）。
- side_by_side.png : ズレが大きい場合の説明用（A|B横並び）。
- overlay_alpha.png: ズレが大きい場合の説明用（Aの上にBを半透明合成）。

============================
■ アルゴリズム概要
============================
1) **カメラ位置のズレ判定**
   - 位相相関（translation）で平行移動量 (dx, dy) を推定し、その大きさが所定閾値以下か確認。
   - 追加で ORB 特徴点 + RANSAC による **Homography** を推定し、
     それが "恒等変換に近い"（平行移動・回転・スケール・遠近成分が小さい）かをチェック。
   → どちらかでズレが大きいと分かれば "ズレあり" と判定。

2) **差分可視化（赤枠方式）**
   - グレースケールの絶対差 `absdiff` をとる。
   - 大津の二値化（または固定閾値）で差分マスクを得る。
   - 小ノイズをモルフォロジー（Open + Dilate）で抑える。
   - 輪郭抽出→外接矩形を A 上に赤枠で描画して `diff_boxes.png` を保存。
   - 差分マスク `diff_mask.png` も保存。

============================
■ パラメータ調整の目安
============================
- `shift_px_thresh`: 位相相関に基づく平行移動量の閾値（px）。デフォルト6px。
  値を小さくすると厳しく（ズレ判定が出やすく）なり、大きくすると緩くなります。
- Homography の近似恒等チェック：
  - 平行移動 |tx|,|ty| ≤ 6.0 px
  - 回転 ≤ 2.0°
  - スケール 1.0±0.05
  - 透視項 |h31|,|h32| ≤ 0.002
  固定カメラで同一構図を想定した、やや厳しめの設定です。
- 差分側の最小サイズ `min_wh`：外接矩形の最小幅・高さ（px）。
  小さなノイズを無視するために 15px を初期値にしています。
- 固定閾値 `bin_thresh` を与えると大津法ではなく固定値で二値化します。

============================
■ 失敗しやすいケースと対策
============================
- 露出・ホワイトバランスが大きく変わる場合 → 二値化が過剰検出になりがち。固定閾値や前処理（平滑化）を調整。
- 画像の一部がクロップ・回転されている場合 → ズレ判定で弾かれます（想定どおり）。
- 画質が粗い/圧縮ノイズが多い場合 → モルフォロジーのカーネルを大きくする、`min_wh` を上げる等で抑制。

============================
■ 使い方
============================
    python make_ref.py [imgA] [imgB]
引数が無い場合はカレントの `test1a.png`, `test1b.png` を使用します。

============================
■ 典型的な標準出力
============================
- カメラ位置が違うと判定された場合：
    カメラの位置が違う画像です。差分処理はスキップします。
    位相相関推定シフト: dx=..., dy=..., norm=...px, H近似恒等=...

- 構図が同じと判定され差分を出した場合：
    カメラ位置はおおむね同じと判定 → 差分を出力しました。
    差分領域の面積比: ...%
    検出ボックス数: N
    最大ボックス: x=..., y=..., w=..., h=...

============================
■ 拡張のヒント（必要に応じて）
============================
- 右側/特定領域のみを見る **ROI制限** を入れる。
- 照明差に強い **SSIM差分** を併用する。
- ズレがある場合に自動で **画像位置合わせ**（ホモグラフィ/ECC）を行ってから差分へ進む。
- 検出した差分領域の色分布を解析して「新規に現れた色」を報告する等の付加機能。

依存:
    pip install opencv-python scikit-image numpy
"""

import os
import sys
import math
import cv2
import numpy as np
import json

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ ユーティリティ ------------------

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"画像が読めません: {path}")
    return img

def resize_to(img, size):
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)

def ensure_same_size(a, b):
    if a.shape[:2] != b.shape[:2]:
        b = resize_to(b, a.shape[:2])
    return a, b

# ------------------ カメラ位置ズレ判定 ------------------

def phase_correlation_shift(g1, g2):
    """
    位相相関（Phase Correlation）で平行移動量 (dx, dy) を推定します。
    ハニング窓で周辺の不連続を和らげ、サブピクセル精度の推定値を返します。
    戻り値は (dx, dy) [px]。ノルムが `shift_px_thresh` 以下なら「ほぼ同位置」とみなします。
    """
    g1f = g1.astype(np.float32)
    g2f = g2.astype(np.float32)
    win = cv2.createHanningWindow((g1.shape[1], g1.shape[0]), cv2.CV_32F)
    s = cv2.phaseCorrelate(g1f * win, g2f * win)[0]  # (dx, dy)
    return s  # (dx, dy)

def estimate_homography(a_gray, b_gray):
    """
    ORB 特徴点 → KNN マッチング（比率テスト 0.75）→ RANSAC で Homography を推定します。
    十分な対応点が得られない/推定に失敗した場合は None を返します。
    """
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
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H

def is_near_identity_homography(H, tol_trans=6.0, tol_rot_deg=2.0, tol_scale=0.05, tol_persp=0.002):
    """
    推定された Homography が恒等変換に“ほぼ等しい”かを簡易チェックします。
    平行移動・回転角・スケール・透視項の4種類をしきい値で評価します。
    これらを超えた場合は「ズレあり」と判定する根拠になります。
    """
    if H is None:
        return False
    # 正規化
    H = H / H[2,2]
    tx, ty = H[0,2], H[1,2]
    if abs(tx) > tol_trans or abs(ty) > tol_trans:
        return False

    # 2x2部分から回転・スケール推定（簡易）
    a,b,c,d = H[0,0], H[0,1], H[1,0], H[1,1]
    # 回転角
    rot_rad = math.atan2(c, a)
    rot_deg = abs(rot_rad * 180.0 / math.pi)
    if rot_deg > tol_rot_deg:
        return False
    # スケール
    sx = math.sqrt(a*a + c*c)
    sy = math.sqrt(b*b + d*d)
    if not (1.0 - tol_scale <= sx <= 1.0 + tol_scale): return False
    if not (1.0 - tol_scale <= sy <= 1.0 + tol_scale): return False

    # 透視項（遠近の歪み）
    if abs(H[2,0]) > tol_persp or abs(H[2,1]) > tol_persp:
        return False

    return True

def camera_misaligned(imgA, imgB,
                      shift_px_thresh=6.0,
                      homography_checks=True):
    """
    カメラ位置がズレているかを総合判定します。
    (1) 位相相関の平行移動量のノルム > しきい値 でズレあり。
    (2) そうでなければ Homography を推定し、恒等近似チェックに通るか判定します。
    戻り値: (misaligned: bool, diagnostics: dict)
      diagnostics には推定シフト、ノルム、Homography の可否が入ります。
    """
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # 1) 位相相関（translation）
    dx, dy = phase_correlation_shift(gA, gB)
    shift_norm = math.hypot(dx, dy)
    if shift_norm > shift_px_thresh:
        return True, {"phase_shift": (dx, dy), "shift_norm": shift_norm, "H_ok": None}

    # 2) Homography近似恒等？
    if homography_checks:
        H = estimate_homography(gA, gB)
        H_ok = is_near_identity_homography(H)
        if not H_ok:
            return True, {"phase_shift": (dx, dy), "shift_norm": shift_norm, "H_ok": False}
        return False, {"phase_shift": (dx, dy), "shift_norm": shift_norm, "H_ok": True}

    return False, {"phase_shift": (dx, dy), "shift_norm": shift_norm, "H_ok": None}

# ------------------ 差分（ズレなし前提） ------------------

def difference_boxes(imgA, imgB, min_wh=15, bin_thresh=None):
    """
    赤枠で差分領域を囲んだ可視化画像と、二値マスク、矩形一覧を返します。
    手順: absdiff → 二値化（大津 or 固定）→ 3x3 開閉/膨張 → 輪郭抽出 → サイズフィルタ。
    `min_wh` は矩形の最小幅・高さ（px）。小ノイズ除去のために利用します。
    """
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gA, gB)

    if bin_thresh is None:
        # Otsuで自動
        _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(diff, bin_thresh, 255, cv2.THRESH_BINARY)

    # ノイズ除去＆連結
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.dilate(th, k, iterations=1)

    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts)==2 else cnts[1]

    vis = imgA.copy()
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < min_wh and h < min_wh:
            continue
        boxes.append((x,y,w,h))
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 2)

    return vis, th, boxes


# ------------------ 追加可視化：赤枠内へBを半透明合成 ------------------


def make_boxes_overlay_transparent(imgA, imgB, boxes, alpha=0.5, draw_border=True):
    """
    赤枠(boxes)で指定された領域に、B側の同位置パッチを半透明(alpha)で重ねる可視化を行う。
    - imgA: 表示のベース（A）
    - imgB: 貼り付け元（B）
    - boxes: [(x,y,w,h), ...]
    - alpha: 0.0(貼らない)～1.0(完全にB) の混合率
    - draw_border: Trueなら枠も描く（赤）
    返り値: 合成済み画像
    """
    out = imgA.copy()
    for (x, y, w, h) in boxes:
        # 範囲安全化
        x = max(0, int(x)); y = max(0, int(y))
        w = max(1, int(w)); h = max(1, int(h))
        x2 = min(out.shape[1], x + w)
        y2 = min(out.shape[0], y + h)
        if x2 <= x or y2 <= y:
            continue

        roiA = out[y:y2, x:x2]
        roiB = imgB[y:y2, x:x2]
        # サイズ違いの保険
        if roiA.shape[:2] != roiB.shape[:2]:
            roiB = cv2.resize(roiB, (roiA.shape[1], roiA.shape[0]), interpolation=cv2.INTER_AREA)

        blended = cv2.addWeighted(roiA, 1.0 - alpha, roiB, alpha, 0.0)
        out[y:y2, x:x2] = blended

        if draw_border:
            cv2.rectangle(out, (x, y), (x2, y2), (0, 0, 255), 2)

    return out

# ------------------ 赤枠が密集する領域をクラスタ化して大きい枠で囲む ------------------

def _boxes_to_mask(boxes, shape, thickness=2):
    """与えられた boxes を真っ黒なマスクに描画して返す（白=枠）。"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, thickness)
    return mask


def cluster_dense_boxes(boxes, shape, dilate_iter=3, min_count=5, min_wh=25):
    """
    赤枠（boxes）が密集して重なる領域をクラスタとして抽出し、
    その外接矩形（大枠）リストを返す。
      - dilate_iter: 膨張回数（距離のゆるさ）
      - min_count  : その大枠内に含まれる元ボックス個数の下限
      - min_wh     : 大枠の最小サイズ（どちらかが未満なら除外）
    """
    if not boxes:
        return []
    h, w = shape[:2]
    # 1) 枠線マスクを作って膨張 → 連結成分を抽出
    mask = _boxes_to_mask(boxes, (h, w), thickness=2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dense = mask.copy()
    for _ in range(max(1, dilate_iter)):
        dense = cv2.dilate(dense, k, iterations=1)
    # 2) 輪郭から大枠を作る
    cnts = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    clusters = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_wh or bh < min_wh:
            continue
        # 3) その枠内に入る元ボックス数を数える（中心点で判定）
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


# ------------------ 大枠のみ描画（小枠なし） ------------------
def draw_clusters_only(base_img, clusters, color=(0, 0, 255), thickness=6, show_count=True, blend_with=None, alpha=0.35):
    """大枠（clusters）だけを描く画像を返す。小さい枠は描かない。
    - base_img: ベースにする画像（通常はA）
    - clusters: [(x,y,w,h,cnt), ...]
    - blend_with: 画像Bを与えると、全体を少しだけBとブレンドして視認性を上げられる
    """
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
    """赤枠密集領域を太い大枠で囲んだ合成画像を作る。"""
    vis = make_boxes_overlay_transparent(base_img, overlay_img, boxes, alpha=alpha, draw_border=True)
    clusters = cluster_dense_boxes(boxes, base_img.shape, dilate_iter=dilate_iter, min_count=min_count, min_wh=min_wh)
    for (x, y, w, h, cnt) in clusters:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        if show_count:
            cv2.putText(vis, f"x{cnt}", (x + 6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis, clusters

# ------------------ ズレあり時の説明用可視化 ------------------

def save_side_by_side(imgA, imgB, out_path):
    h = max(imgA.shape[0], imgB.shape[0])
    w = imgA.shape[1] + imgB.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:imgA.shape[0], :imgA.shape[1]] = imgA
    canvas[:imgB.shape[0], imgA.shape[1]:imgA.shape[1]+imgB.shape[1]] = imgB
    cv2.imwrite(out_path, canvas)


def save_alpha_overlay(imgA, imgB, out_path, alpha=0.35):
    b = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]), interpolation=cv2.INTER_AREA)
    blended = cv2.addWeighted(imgA, 1.0 - alpha, b, alpha, 0.0)
    cv2.imwrite(out_path, blended)

# ------------------ 整列（ECC → Homography フォールバック） ------------------

def align_ecc(a, b):
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2,3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
        cc, warp = cv2.findTransformECC(ag, bg, warp, cv2.MOTION_AFFINE, criteria)
        aligned = cv2.warpAffine(b, warp, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        return aligned
    except cv2.error:
        return None


def align_homography(a, b):
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(ag, None)
    k2, d2 = orb.detectAndCompute(bg, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p for p,q in m if p.distance < 0.75*q.distance]
    if len(good) < 12:
        return None
    src = np.float32([k1[p.queryIdx].pt for p in good]).reshape(-1,1,2)
    dst = np.float32([k2[p.trainIdx].pt for p in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    if H is None:
        return None
    return cv2.warpPerspective(b, H, (a.shape[1], a.shape[0]))

# ------------------ 照明差に強い差分（勾配 + Census） ------------------

def grad_mag(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1,0,ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0,1,ksize=3)
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
            mask = ((cmp << bit) & 0xFF).astype(np.uint8)
            out = cv2.bitwise_or(out, mask)
            bit = (bit + 1) % 8
    return out


def fused_diff_mask(a, b):
    d1 = cv2.absdiff(grad_mag(a), grad_mag(b))
    d2 = cv2.absdiff(census8(a), census8(b))
    fused = cv2.addWeighted(d1, 0.6, d2, 0.4, 0)
    # 適応二値化（窓サイズは短辺に応じて自動）
    h, w = fused.shape
    ks = max(15, (min(h, w)//40)|1)
    th = cv2.adaptiveThreshold(fused, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ks, -3)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    return th


def boxes_from_mask(imgA, mask, min_wh=15):
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

# ------------------ ドア判定（明るさ非依存・任意実行） ------------------

def door_roi(img, fallback_ratio=(0.65, 1.0, 0.20, 0.95)):
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90, minLineLength=h//4, maxLineGap=10)
    if lines is not None:
        xs = []
        for (x1,y1,x2,y2) in lines[:,0]:
            dx, dy = abs(x2-x1), abs(y2-y1)
            slope_ok = (dx == 0) or ((dy/(dx+1e-6)) > 2.5)
            if slope_ok and max(x1,x2) > w*0.55:
                xs += [x1, x2]
        if len(xs) >= 2:
            x1 = max(0, int(min(xs) - 10)); x2 = min(w, int(max(xs) + 10))
            y1 = int(0.20*h); y2 = int(0.95*h)
            if x2-x1 > 20 and y2-y1 > 20:
                return (x1, y1, x2-x1, y2-y1)
    # fallback
    x1 = int(fallback_ratio[0]*w); x2 = int(fallback_ratio[1]*w)
    y1 = int(fallback_ratio[2]*h); y2 = int(fallback_ratio[3]*h)
    return (x1, y1, x2-x1, y2-y1)


def _roi(img, box):
    x,y,w,h = box
    return img[y:y+h, x:x+w]


def grad_orient_hist(img, box, nbins=36):
    roi = cv2.cvtColor(_roi(img, box), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(roi, cv2.CV_32F, 1,0,ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0,1,ksize=3)
    mag = cv2.magnitude(gx,gy)
    ang = cv2.phase(gx,gy,angleInDegrees=False)
    ang = np.mod(ang, np.pi)
    bins = (ang / np.pi * nbins).astype(np.int32)
    bins[bins==nbins] = nbins-1
    hist = np.bincount(bins.ravel(), weights=mag.ravel(), minlength=nbins).astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def hough_oblique_ratio(img, box, min_len=20, oblique_deg=(15,75)):
    roi = cv2.cvtColor(_roi(img, box), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(roi, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return 0.0, 0.0
    tot = 0.0; obl = 0.0
    lo, hi = np.deg2rad(oblique_deg[0]), np.deg2rad(oblique_deg[1])
    for (x1,y1,x2,y2) in lines[:,0]:
        dx, dy = x2-x1, y2-y1
        L = float(np.hypot(dx,dy))
        if L < min_len: continue
        ang = abs(np.arctan2(dy,dx)) % np.pi
        tot += L
        if lo <= ang <= hi and not (abs(ang-np.pi/2)<np.deg2rad(3)) and not (ang<np.deg2rad(3) or abs(ang-np.pi)<np.deg2rad(3)):
            obl += L
    return (obl/tot if tot>0 else 0.0), tot


def census_hist(img, box):
    roi = _roi(img, box)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h,w = g.shape
    out = np.zeros((h,w), np.uint8); bit=0
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx==0 and dy==0: continue
            shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
            cmp = (g > shifted).astype(np.uint8)
            out = cv2.bitwise_or(out, ((cmp << bit) & 0xFF))
            bit = (bit+1)%8
    hist = np.bincount(out.ravel(), minlength=256).astype(np.float32)
    if hist.sum()>0: hist/=hist.sum()
    return hist


def chi2(p,q, eps=1e-8):
    d = p-q
    return 0.5*np.sum((d*d)/(p+q+eps))


def orb_h_inlier_ratio(imgA, imgB, box):
    ra = _roi(imgA, box); rb = _roi(imgB, box)
    ga = cv2.cvtColor(ra, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(rb, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1500)
    k1,d1 = orb.detectAndCompute(ga,None)
    k2,d2 = orb.detectAndCompute(gb,None)
    if d1 is None or d2 is None: return 0.0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
    good = [p for p,q in m if p.distance < 0.75*q.distance]
    if len(good) < 10: return 0.0
    src = np.float32([k1[p.queryIdx].pt for p in good]).reshape(-1,1,2)
    dst = np.float32([k2[p.trainIdx].pt for p in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 4.0)
    if H is None or mask is None: return 0.0
    return float(mask.sum())/len(mask)


def draw_door_vis(base, box, score, is_open):
    x,y,w,h = box
    vis = base.copy()
    color = (0,255,0) if is_open else (0,0,255)
    cv2.rectangle(vis, (x,y), (x+w, y+h), color, 2)
    cv2.putText(vis, f"DOOR_OPEN={is_open} score={score:.2f}", (x+5, max(15,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


def score_door_open_lightless(imgA, imgB, box, thr=0.60):
    hA = grad_orient_hist(imgA, box, nbins=36)
    hB = grad_orient_hist(imgB, box, nbins=36)
    sim = float(np.dot(hA, hB) / (np.linalg.norm(hA)*np.linalg.norm(hB)+1e-8))
    s_orient = max(0.0, min(1.0, 1.0 - sim))

    rA,_ = hough_oblique_ratio(imgA, box)
    rB,_ = hough_oblique_ratio(imgB, box)
    s_oblique = max(0.0, min(1.0, abs(rA - rB) * 2.0))

    cA = census_hist(imgA, box)
    cB = census_hist(imgB, box)
    s_census = max(0.0, min(1.0, chi2(cA, cB) * 2.0))

    inlier = orb_h_inlier_ratio(imgA, imgB, box)
    s_plane = max(0.0, min(1.0, 1.0 - inlier))

    score = 0.35*s_orient + 0.25*s_oblique + 0.20*s_census + 0.20*s_plane
    is_open = (score >= thr)
    return score, is_open, {
        "orient_1msim": s_orient,
        "oblique_ratio_delta": abs(rA-rB),
        "census_chi2": float(chi2(cA,cB)),
        "plane_inlier_ratio": float(inlier)
    }

# ------------------ JSON セーフ変換 ------------------

def _to_jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, bool):
        return o
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    return o

def main():
    # 入力
    if len(sys.argv) >= 3:
        pathA, pathB = sys.argv[1], sys.argv[2]
    else:
        pathA, pathB = "test1a.png", "test1b.png"

    imgA = read_image(pathA)
    imgB = read_image(pathB)
    imgA, imgB = ensure_same_size(imgA, imgB)

    # カメラ位置ズレ判定
    misaligned, diag = camera_misaligned(imgA, imgB,
                                         shift_px_thresh=6.0,
                                         homography_checks=True)

    if misaligned:
        print("カメラの位置が違う画像です。差分処理はスキップします。")
        print(f"  位相相関推定シフト: dx={diag['phase_shift'][0]:.2f}, dy={diag['phase_shift'][1]:.2f}, "
              f"norm={diag['shift_norm']:.2f}px, H近似恒等={diag['H_ok']}")
        # 説明用出力
        save_side_by_side(imgA, imgB, os.path.join(OUT_DIR, "side_by_side.png"))
        save_alpha_overlay(imgA, imgB, os.path.join(OUT_DIR, "overlay_alpha.png"), alpha=0.35)
        print("  説明用: side_by_side.png / overlay_alpha.png を保存しました。")

        # 自動整列トライ（成功したら続行）
        aligned_method = None
        b_ecc = align_ecc(imgA, imgB)
        if b_ecc is not None:
            print("  ECC整列に成功し、差分処理を継続します。")
            imgB = b_ecc
            aligned_method = "ECC"
        else:
            b_h = align_homography(imgA, imgB)
            if b_h is not None:
                print("  Homography整列に成功し、差分処理を継続します。")
                imgB = b_h
                aligned_method = "H"

        if aligned_method is None:
            print("  自動整列にも失敗したため、ここで終了します。")
            return
    else:
        aligned_method = None

    # （ここから先は“構図が同じ”前提の差分可視化）
    strict_same = (aligned_method is None) and (diag.get("shift_norm", 0.0) <= 2.0)

    if strict_same:
        vis_boxes, th_mask, boxes = difference_boxes(imgA, imgB, min_wh=15, bin_thresh=None)
    else:
        th_mask = fused_diff_mask(imgA, imgB)
        vis_boxes, boxes = boxes_from_mask(imgA, th_mask, min_wh=15)

    cv2.imwrite(os.path.join(OUT_DIR, "diff_mask.png"), th_mask)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_boxes.png"), vis_boxes)

    if boxes:
        vis_boxesx = make_boxes_overlay_transparent(imgA, imgB, boxes, alpha=0.5, draw_border=True)
        cv2.imwrite(os.path.join(OUT_DIR, "diff_boxesx.png"), vis_boxesx)
        # diff_boxesx2.png: 大きい枠のみ（小さい枠は描かない）
        clusters = cluster_dense_boxes(
            boxes, imgA.shape,
            dilate_iter=3,   # 近接枠のつながり具合（3~5で調整）
            min_count=6,     # 何個以上重なれば“大枠”とみなすか
            min_wh=30        # 小さすぎる大枠は無視
        )
        vis_boxesx2 = draw_clusters_only(
            imgA, clusters,
            color=(0, 0, 255), thickness=6, show_count=True,
            blend_with=None  # test1a(imgA)の上に同じ場所で大枠のみ描画
        )
        cv2.imwrite(os.path.join(OUT_DIR, "diff_boxesx2.png"), vis_boxesx2)

    area_ratio = (th_mask>0).sum() / float(th_mask.size)
    print("カメラ位置はおおむね同じと判定 → 差分を出力しました。")
    print(f"  差分領域の面積比: {area_ratio*100:.2f}%")
    print(f"  検出ボックス数: {len(boxes)}")
    if boxes:
        x,y,w,h = max(boxes, key=lambda b: b[2]*b[3])
        print(f"  最大ボックス: x={x}, y={y}, w={w}, h={h}")
    if misaligned and aligned_method is not None:
        print(f"  画像整列: {aligned_method}")

    try:
        ans = input("ドア判定を行いますか？ (y/n): ").strip().lower()
    except EOFError:
        ans = 'n'
    if ans == 'y':
        box = door_roi(imgA)
        score, is_open, detail = score_door_open_lightless(imgA, imgB, box, thr=0.60)
        door_vis = draw_door_vis(imgA, box, score, is_open)
        cv2.imwrite(os.path.join(OUT_DIR, "door_vis.png"), door_vis)
        report = {
            "alignment_method": ("None" if aligned_method is None else aligned_method),
            "phase_shift": {"dx": diag.get("phase_shift", (0,0))[0], "dy": diag.get("phase_shift", (0,0))[1], "norm": diag.get("shift_norm", 0.0)},
            "H_ok": diag.get("H_ok", None),
            "door_score_lightless": score,
            "door_is_open": is_open,
            "door_detail": detail
        }
        with open(os.path.join(OUT_DIR, "door_report.json"), "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(report), f, ensure_ascii=False, indent=2)
        print(f"  ドア判定スコア(明るさ非依存): {score:.2f} → 開いていると判定: {is_open}")
    else:
        print("ドア判定はスキップしました。")

if __name__ == "__main__":
    main()