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
        return

    # （ここから先は“構図が同じ”前提の差分可視化）
    # 1) 赤枠方式
    vis_boxes, th_mask, boxes = difference_boxes(imgA, imgB, min_wh=15, bin_thresh=None)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_boxes.png"), vis_boxes)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_mask.png"), th_mask)

    # 追加出力: 赤枠内にBを半透明で重ねた“見比べやすい”画像
    if boxes:
        vis_boxesx = make_boxes_overlay_transparent(imgA, imgB, boxes, alpha=0.5, draw_border=True)
        cv2.imwrite(os.path.join(OUT_DIR, "diff_boxesx.png"), vis_boxesx)

    # ざっくり要約
    area_ratio = (th_mask>0).sum() / float(th_mask.size)
    print("カメラ位置はおおむね同じと判定 → 差分を出力しました。")
    print(f"  差分領域の面積比: {area_ratio*100:.2f}%")
    print(f"  検出ボックス数: {len(boxes)}")
    if boxes:
        x,y,w,h = max(boxes, key=lambda b: b[2]*b[3])
        print(f"  最大ボックス: x={x}, y={y}, w={w}, h={h}")

if __name__ == "__main__":
    main()