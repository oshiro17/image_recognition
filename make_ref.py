# make_ref.py
# -*- coding: utf-8 -*-
"""
カメラ位置ズレ検出 →（必要なら自動整列）→ 画質整合 → 重なりマスク → 勾配+Census/SSIM 差分
を行い、可視化と指標を出力するスクリプト。

出力（results/ 配下）:
- diff_boxes.png    : A上に赤枠で差分領域を描画
- diff_mask.png     : 差分の二値マスク
- diff_boxesx.png   : 赤枠内へBを半透明で重ねた可視化
- diff_boxesx2.png  : 密集領域を大枠のみで描画（ある場合）
- side_by_side.png  : ズレ時の説明用（A|B横並び）
- overlay_alpha.png : ズレ時の説明用（A上にB半透明）

使い方:
    python make_ref.py [imgA] [imgB]
（引数が無ければ test1a.png / test1b.png を使用）
"""

import os
import sys
import cv2
import numpy as np

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- モジュール化したコア関数群 -------------------------------------------------
from core.io_utils import read_image, ensure_same_size, save_side_by_side, save_alpha_overlay
from core.align    import camera_misaligned, align_ecc, align_homography
from core.quality  import quality_harmonize_pair
from core.overlap  import overlap_mask, masked_fused_diff
from core.diff     import boxes_from_mask, ssim_on_overlap
from core.vis      import make_boxes_overlay_transparent, draw_clusters_only
from core.cluster  import cluster_dense_boxes


def main():
    # 入力
    if len(sys.argv) >= 3:
        pathA, pathB = sys.argv[1], sys.argv[2]
    else:
        pathA, pathB = "test1a.png", "test1b.png"

    imgA = read_image(pathA)
    imgB = read_image(pathB)
    imgA, imgB = ensure_same_size(imgA, imgB)

    # 1) カメラ位置ズレ判定
    misaligned, diag = camera_misaligned(imgA, imgB,
                                         shift_px_thresh=6.0,
                                         homography_checks=True)

    tfm_kind, tfm = None, None
    aligned_method = None

    if misaligned:
        print("カメラの位置が違う画像です。差分処理はスキップします。")
        print(f"  位相相関推定シフト: dx={diag['phase_shift'][0]:.2f}, dy={diag['phase_shift'][1]:.2f}, "
              f"norm={diag['shift_norm']:.2f}px, H近似恒等={diag['H_ok']}")
        # 説明用出力
        save_side_by_side(imgA, imgB, os.path.join(OUT_DIR, "side_by_side.png"))
        save_alpha_overlay(imgA, imgB, os.path.join(OUT_DIR, "overlay_alpha.png"), alpha=0.35)
        print("  説明用: side_by_side.png / overlay_alpha.png を保存しました。")

        # 2) 自動整列（ECC → H）
        b_ecc, warp = align_ecc(imgA, imgB)
        if b_ecc is not None:
            print("  ECC整列に成功し、差分処理を継続します。")
            imgB = b_ecc
            aligned_method = "ECC"
            tfm_kind, tfm = "AFFINE", warp
        else:
            b_h, H = align_homography(imgA, imgB)
            if b_h is not None:
                print("  Homography整列に成功し、差分処理を継続します。")
                imgB = b_h
                aligned_method = "H"
                tfm_kind, tfm = "H", H
            else:
                print("  自動整列にも失敗したため、ここで終了します。")
                return

    # 3) 画質整合（解像・シャープ・明るさ）→ 同倍率で両者を縮小・整合
    imgA, imgB = quality_harmonize_pair(
        imgA, imgB,
        target_short=720,   # 短辺720px にそろえる（比較安定化）
        match_sharp=True,   # シャープネスの差を軽く矯正
        match_hist=True,    # 明るさ/コントラストを近づける
        denoise='none'      # 必要なら 'light' / 'add_noise'
    )
    imgA, imgB = ensure_same_size(imgA, imgB)

    # 4) 重なり領域のマスク（整列行列があればそれを使用）
    overlap = overlap_mask(
        imgA.shape,
        tfm,
        ("AFFINE" if tfm_kind == "AFFINE" else "H") if tfm is not None else None,
        imgB.shape
    )

    # 5) 勾配+Census の融合差分 → 重なり外は0
    th_mask = masked_fused_diff(imgA, imgB, overlap)

    # 6) 可視化：赤枠描画 / 半透明合成 / 密集領域の大枠
    vis_boxes, boxes = boxes_from_mask(imgA, th_mask, min_wh=15)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_mask.png"), th_mask)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_boxes.png"), vis_boxes)

    if boxes:
        vis_boxesx = make_boxes_overlay_transparent(imgA, imgB, boxes, alpha=0.5, draw_border=True)
        cv2.imwrite(os.path.join(OUT_DIR, "diff_boxesx.png"), vis_boxesx)

        clusters = cluster_dense_boxes(
            boxes, imgA.shape,
            dilate_iter=3,  # 近接枠をどれくらい繋げるか
            min_count=6,    # いくつ以上重なれば“大枠”
            min_wh=30       # 大枠の最小サイズ
        )
        vis_boxesx2 = draw_clusters_only(
            imgA, clusters,
            color=(0, 0, 255), thickness=6, show_count=True,
            blend_with=None
        )
        cv2.imwrite(os.path.join(OUT_DIR, "diff_boxesx2.png"), vis_boxesx2)

    # 7) 指標: 重なり率・SSIM（重なり内）・差分面積比
    overlap_area  = (overlap > 0).sum()
    diff_area     = ((th_mask > 0) & (overlap > 0)).sum()
    area_ratio    = diff_area / float(overlap_area + 1e-8)
    ss            = ssim_on_overlap(imgA, imgB, overlap)
    overlap_ratio = overlap_area / float(overlap.size)

    # 判定しきい値
    SAME_OVERLAP_THR = 0.35     # 重なり35%以上
    SAME_SSIM_THR    = 0.92     # SSIM>=0.92
    SAME_DIFF_THR    = 0.005    # 差分面積比<=0.5%

    same_scene = (
        overlap_ratio >= SAME_OVERLAP_THR
        and (ss is not None and ss >= SAME_SSIM_THR)
        and (area_ratio <= SAME_DIFF_THR)
    )

    if same_scene:
        print("同一風景と判定（差分なし）")
    else:
        print("差分を出力しました。")
    print(f"  オーバーラップ率: {overlap_ratio*100:.2f}%")
    print(f"  SSIM(重なり内): {('None' if ss is None else f'{ss:.4f}')}")
    print(f"  差分領域の面積比(重なり内): {area_ratio*100:.2f}%")
    print(f"  検出ボックス数: {len(boxes)}")
    if boxes:
        x,y,w,h = max(boxes, key=lambda b: b[2]*b[3])
        print(f"  最大ボックス: x={x}, y={y}, w={w}, h={h}")
    if misaligned and aligned_method is not None:
        print(f"  画像整列: {aligned_method}")


if __name__ == "__main__":
    main()