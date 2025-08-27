# app.py
# -*- coding: utf-8 -*-
"""
📷 基準撮影 → 設定 → 監視 の3ステップUI
- 起動直後は「基準を撮影/読み込み」のみ表示
- 基準確定後に設定（撮影間隔・整列など）
- スタートでループ（基準 vs 現在）差分をリアルタイム表示
- すべての撮影＆差分は runs/<session>/ に保存、履歴ギャラリーで見返し＆ZIP DL
"""

import os
import io
import glob
import time
import json
import shutil
import cv2
import numpy as np
import streamlit as st
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# 自動リロード（存在すればpkg、なければJSフォールバック）
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_IMPL = "pkg"
except Exception:
    from streamlit.components.v1 import html as _html
    def st_autorefresh(interval: int = 5_000, key: str = "tick", limit: int | None = None):
        _html(f"<script>setTimeout(()=>window.location.reload(), {interval});</script>", height=0)
        return None
    _AUTOREFRESH_IMPL = "js"

# ---- 既存 core モジュール ----
from core.io_utils import ensure_same_size
from core.align import camera_misaligned, align_ecc, align_homography
# from core.diff import difference_boxes, fused_diff_mask, boxes_from_mask, quality_harmonize_pair
from core.vis import make_boxes_overlay_transparent, draw_clusters_only
from core.cluster import cluster_dense_boxes

from core.diff import difference_boxes, fused_diff_mask, boxes_from_mask
from core.quality import quality_harmonize_pair

# ============ 共通UI ============

st.set_page_config(page_title="差分監視 📸", layout="wide")
st.markdown(
    """
    <style>
      .big-badge {font-size: 20px; padding: 6px 14px; border-radius: 999px; display:inline-block;}
      .ok-badge  {background:#eafaf1; color:#1f9254; border:1px solid #b8e6cd;}
      .ng-badge  {background:#ffefef; color:#c0392b; border:1px solid #ffd0d0;}
      .pill      {background:#f3f6ff; color:#3b5bcc; border:1px solid #d9e2ff; border-radius:999px; padding:2px 10px; font-size:12px;}
      .card {padding:14px; border-radius:14px; border:1px solid #eee; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.03);}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🌸 差分監視")

# ============ セッション状態 ============

ss = st.session_state
def _init_state():
    ss.setdefault("phase", "INIT")          # INIT -> BASELINE_SET -> CONFIG_SET -> RUNNING/PAUSED
    ss.setdefault("session_dir", "")        # runs/<session_name>
    ss.setdefault("baseline_path", "")      # 保存先
    ss.setdefault("config", {})             # 設定保存
    ss.setdefault("next_shot_ts", 0.0)      # 次回撮影時刻
    ss.setdefault("cam_diag", {})           # カメラ情報
    ss.setdefault("last_metrics", None)     # 直近メトリクス
    ss.setdefault("backend", "AVFOUNDATION" if sys.platform == "darwin" else "AUTO")
    ss.setdefault("cam_index", 0)

_init_state()

# ============ ラン（保存先） ============

def new_session_dir() -> str:
    root = Path("runs")
    root.mkdir(exist_ok=True)
    name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    session = root / name
    (session / "captures").mkdir(parents=True, exist_ok=True)
    (session / "diffs").mkdir(parents=True, exist_ok=True)
    return str(session)

def save_config(session_dir: str, cfg: Dict[str, Any]) -> None:
    with open(Path(session_dir)/"config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def append_index(session_dir: str, row: Dict[str, Any]) -> None:
    import csv
    idx = Path(session_dir)/"index.csv"
    new = not idx.exists()
    with open(idx, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts","capture","aligned","ssim","diff_ratio","boxes"])
        if new: w.writeheader()
        w.writerow(row)

def make_zip(session_dir: str) -> str:
    base = Path(session_dir).resolve()
    out = base.parent / (base.name + ".zip")
    if out.exists():
        out.unlink()
    shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=str(base))
    return str(out)

# ============ カメラユーティリティ ============

def _backend_flag(name: str) -> Optional[int]:
    m = {
        "AVFOUNDATION": getattr(cv2, "CAP_AVFOUNDATION", None),
        "QT": getattr(cv2, "CAP_QT", None),
        "V4L2": getattr(cv2, "CAP_V4L2", None),
        "DSHOW": getattr(cv2, "CAP_DSHOW", None),
        "AUTO": None,
    }
    return m.get((name or "AUTO").upper())

def capture_frame(index: int, backend: str, warmup: int = 5, size: Tuple[int,int] | None=None) -> Tuple[Optional[np.ndarray], Dict[str,Any]]:
    order = [backend] + [b for b in (("AVFOUNDATION","QT") if sys.platform=="darwin" else ("V4L2","DSHOW")) if b!=backend] + (["AUTO"] if backend!="AUTO" else [])
    for b in order:
        flag = _backend_flag(b)
        cap = cv2.VideoCapture(index, flag) if flag is not None else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release(); continue
        if size:
            w,h = size; cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(max(0,warmup)): cap.read()
        ok, frame = cap.read(); cap.release()
        if ok and isinstance(frame, np.ndarray) and frame.size>0:
            return frame, {"backend_used": b, "shape": tuple(frame.shape)}
    return None, {"backend_used": None, "shape": None}

def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ============ 差分処理（基準 vs 現在） ============

def compare_to_baseline(baseline_bgr: np.ndarray, current_bgr: np.ndarray, cfg: Dict[str,Any]) -> Dict[str,Any]:
    # 画質整合（同倍率ダウンサンプル・シャープ/明るさ合わせ）
    a, b = quality_harmonize_pair(baseline_bgr, current_bgr,
                                  target_short=cfg.get("target_short", 720),
                                  match_sharp=True, match_hist=True, denoise='none')
    a, b = ensure_same_size(a, b)

    # ズレ判定→整列（任意）
    aligned_method = "NONE"
    if cfg.get("align_mode","AUTO") != "OFF":
        misaligned, _d = camera_misaligned(a, b, shift_px_thresh=cfg.get("shift_px_thresh", 6.0), homography_checks=True)
        if misaligned:
            if cfg.get("align_mode","AUTO") in ("AUTO","ECC"):
                b_ecc, _warp = align_ecc(a, b)
                if b_ecc is not None:
                    b = b_ecc; aligned_method = "ECC"
            if aligned_method == "NONE" and cfg.get("align_mode","AUTO") in ("AUTO","H"):
                b_h, _H = align_homography(a, b)
                if b_h is not None:
                    b = b_h; aligned_method = "H"

    # 厳密一致ならabsdiff赤枠、そうでなければ照明に強いマスク
    vis_boxes, th_mask, boxes = None, None, []
    strict_same = (aligned_method=="NONE")
    if strict_same:
        vis_boxes, th_mask, boxes = difference_boxes(a, b, min_wh=cfg.get("min_wh", 15), bin_thresh=None)
    else:
        th_mask = fused_diff_mask(a, b)
        vis_boxes, boxes = boxes_from_mask(a, th_mask, min_wh=cfg.get("min_wh", 15))

    diff_ratio = float((th_mask>0).sum()) / float(th_mask.size)
    overlay = make_boxes_overlay_transparent(a, b, boxes, alpha=0.5, draw_border=True) if boxes else a.copy()
    clusters = cluster_dense_boxes(boxes, a.shape, dilate_iter=3, min_count=6, min_wh=30)
    clusters_vis = draw_clusters_only(a, clusters, color=(0,0,255), thickness=6, show_count=True)

    # SSIM（マスクなし簡易）
    try:
        from skimage.metrics import structural_similarity as ssim
        ss = ssim(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    except Exception:
        ss = None

    return {
        "A": a, "B": b,
        "vis_boxes": vis_boxes, "mask": th_mask, "overlay": overlay, "clusters_vis": clusters_vis,
        "boxes": boxes, "diff_ratio": diff_ratio, "ssim": ss,
        "aligned": aligned_method
    }

# ============ UI：ステップ 1（基準） ============

def ui_step_baseline():
    st.header("① 基準を撮影 / 読み込み")
    colL, colR = st.columns(2)

    with colL:
        st.markdown("#### 📸 カメラで撮影")
        idx = st.number_input("カメラデバイス番号", 0, 10, ss.get("cam_index",0), 1, key="k_cam_index")
        be  = st.selectbox("バックエンド", ["AUTO","AVFOUNDATION","QT","V4L2","DSHOW"],
                           index=(1 if sys.platform=="darwin" else 0), key="k_backend")
        if st.button("📷 基準を撮る（クリックで1枚）", use_container_width=True, type="primary"):
            frame, diag = capture_frame(idx, be)
            if frame is None:
                st.error("カメラ取得に失敗しました。デバイス番号/バックエンドを見直してください。")
            else:
                session = new_session_dir()
                base_path = str(Path(session)/"baseline.png")
                cv2.imwrite(base_path, frame)
                ss.session_dir = session
                ss.baseline_path = base_path
                ss.cam_index = idx
                ss.backend = be
                ss.cam_diag = diag
                ss.phase = "BASELINE_SET"
                st.success("基準を保存しました！次へ進めます。")
                st.rerun()

    with colR:
        st.markdown("#### 🖼 画像から読み込む")
        up = st.file_uploader("基準画像（PNG/JPG）", type=["png","jpg","jpeg"])
        if up is not None:
            data = np.frombuffer(up.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                st.error("画像を読み込めませんでした。")
            else:
                session = new_session_dir()
                base_path = str(Path(session)/"baseline.png")
                cv2.imwrite(base_path, img)
                ss.session_dir = session
                ss.baseline_path = base_path
                ss.phase = "BASELINE_SET"
                st.success("基準を保存しました！次へ進めます。")
                st.rerun()

# ============ UI：ステップ 2（設定） ============

def ui_step_config():
    st.header("② 設定")
    with st.form("cfg_form", clear_on_submit=False):
        st.markdown("**撮影**")
        interval = st.slider("撮影間隔（秒）⏱️", 10, 600, 60, 5)
        target_short = st.slider("共通ダウンサンプル短辺（px）", 360, 1440, 720, 60)

        st.markdown("---")
        st.markdown("**整列と検出**")
        align_mode = st.selectbox("整列モード", ["AUTO","ECC","H","OFF"], help="AUTO推奨")
        shift_thresh = st.slider("ズレ判定しきい値(px)", 1.0, 20.0, 6.0, 0.5)
        min_wh = st.slider("最小ボックス幅/高さ(px)", 5, 100, 15, 1)

        st.markdown("---")
        submitted = st.form_submit_button("✅ 設定を保存して次へ", use_container_width=True)
        if submitted:
            ss.config = {
                "interval": int(interval),
                "target_short": int(target_short),
                "align_mode": align_mode,
                "shift_px_thresh": float(shift_thresh),
                "min_wh": int(min_wh),
                "camera": {"index": ss.cam_index, "backend": ss.backend}
            }
            save_config(ss.session_dir, ss.config)
            ss.phase = "CONFIG_SET"
            # 次回撮影時刻はスタート時にセットする（停止中にカウントダウンさせない）
            ss.next_shot_ts = 0.0
            st.success("設定を保存しました！")
            st.rerun()

# ============ UI：ステップ 3（監視） ============

def ui_step_run():
    st.header("③ 監視スタート")
    colA, colB = st.columns([2,1])

    with colB:
        # ステータス＆操作
        running = ss.get("phase") == "RUNNING"
        status_badge = f"<span class='big-badge {'ok-badge' if running else 'ng-badge'}'>{'🟢 監視中' if running else '🔴 停止中'}</span>"
        st.markdown(status_badge, unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns(2)
        if c1.button("▶️ スタート", use_container_width=True):
            ss.phase = "RUNNING"
            # スタート時に次回撮影時刻をセット
            ss.next_shot_ts = time.time() + ss.config["interval"]
            st.rerun()
        if c2.button("⏸ 停止", use_container_width=True):
            ss.phase = "PAUSED"
            st.rerun()

        st.markdown("---")
        # カウントダウン
        if running and ss.get("next_shot_ts", 0) > 0:
            target_ms = int(ss.next_shot_ts * 1000)
            interval_s = int(ss.config.get("interval", 60))
            st.components.v1.html(f"""
            <div style='margin:4px 0 10px 0;'>
              次の撮影まで: <b><span id='cd_secs'>--</span> 秒</b> ⏳
              <div style='height:8px;background:#222;border-radius:6px;overflow:hidden;margin-top:6px;'>
                <div id='cd_bar' style='height:100%;width:0%;background:#3b82f6;'></div>
              </div>
            </div>
            <script>
            const target = {target_ms};
            const interval = {interval_s};
            function tick(){{
              const now = Date.now();
              const remain = Math.max(0, Math.ceil((target - now)/1000));
              document.getElementById('cd_secs').textContent = remain;
              const p = 100 * Math.min(1, Math.max(0, (interval - remain)/interval));
              document.getElementById('cd_bar').style.width = p + '%';
            }}
            tick();
            setInterval(tick, 250);
            </script>
            """, height=60)
        else:
            st.caption("停止中（カウントダウンなし）")

        # ZIP DL
        if st.button("💾 このセッションをZIPでダウンロード", use_container_width=True):
            zpath = make_zip(ss.session_dir)
            with open(zpath, "rb") as f:
                st.download_button("ZIPをダウンロード", f, file_name=Path(zpath).name, mime="application/zip", use_container_width=True)

        st.markdown("---")
        st.caption("基準画像プレビュー")
        base = cv2.imread(ss.baseline_path, cv2.IMREAD_COLOR)
        if base is not None:
            st.image(bgr2rgb(base), use_container_width=True)

    with colA:
        # ライブ差分
        st.subheader("リアルタイム差分")
        ph_cols = st.columns(3)
        phA, phB, phD = ph_cols[0].empty(), ph_cols[1].empty(), ph_cols[2].empty()
        info = st.empty()

        # RUNNING のときだけリフレッシュ（チラつき抑制）
        running = ss.get("phase") == "RUNNING"
        if running:
            st_autorefresh(interval=max(1000, int(ss.config.get("interval", 60) * 1000)), key="tick-live")

        # 条件：RUNNINGなら時刻到達で撮影→差分→保存
        if ss.phase == "RUNNING" and time.time() >= ss.next_shot_ts:
            # 撮影
            frame, diag = capture_frame(ss.config["camera"]["index"], ss.config["camera"]["backend"])
            if frame is not None:
                # 差分
                baseline = cv2.imread(ss.baseline_path, cv2.IMREAD_COLOR)
                res = compare_to_baseline(baseline, frame, ss.config)
                # 保存
                tsname = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                cap_path = str(Path(ss.session_dir)/"captures"/f"{tsname}.jpg")
                diff_path = str(Path(ss.session_dir)/"diffs"/f"{tsname}_diff.png")
                mask_path = str(Path(ss.session_dir)/"diffs"/f"{tsname}_mask.png")
                cv2.imwrite(cap_path, frame)
                cv2.imwrite(diff_path, res["vis_boxes"])
                cv2.imwrite(mask_path, res["mask"])
                append_index(ss.session_dir, {
                    "ts": tsname, "capture": Path(cap_path).name,
                    "aligned": res["aligned"], "ssim": (res["ssim"] if res["ssim"] is not None else -1),
                    "diff_ratio": res["diff_ratio"], "boxes": len(res["boxes"])
                })
                ss.last_metrics = {"ts": tsname, "aligned": res["aligned"], "ssim": res["ssim"], "diff_ratio": res["diff_ratio"], "boxes": len(res["boxes"])}
            # 次回時刻
            ss.next_shot_ts = time.time() + ss.config["interval"]

        # 表示（直近の撮影結果 or 最新の保存済み）
        baseline = cv2.imread(ss.baseline_path, cv2.IMREAD_COLOR)
        try:
            latest_list = sorted(glob.glob(str(Path(ss.session_dir)/"captures/*.jpg")))
            latest_cap = latest_list[-1] if latest_list else None
        except Exception:
            latest_cap = None
        latest_diff = None
        latest_mask = None
        if latest_cap:
            tsname = Path(latest_cap).stem
            latest_diff = Path(ss.session_dir)/"diffs"/f"{tsname}_diff.png"
            latest_mask = Path(ss.session_dir)/"diffs"/f"{tsname}_mask.png"

        if latest_cap and baseline is not None:
            phA.image(bgr2rgb(baseline), caption="基準", use_container_width=True)
            imgB = cv2.imread(latest_cap, cv2.IMREAD_COLOR)
            phB.image(bgr2rgb(imgB), caption="今回ショット", use_container_width=True)
            if latest_diff and latest_mask and latest_diff.exists() and latest_mask.exists():
                grid = st.columns([2,1])
                with grid[0]:
                    phD.image(bgr2rgb(cv2.imread(str(latest_diff))), caption="差分（赤枠）", use_container_width=True)
                with grid[1]:
                    st.image(cv2.imread(str(latest_mask), cv2.IMREAD_GRAYSCALE), clamp=True, caption="差分マスク", use_container_width=True)

        lm = ss.get("last_metrics") or {}
        info.info(f"整列: {lm.get('aligned','-')} / SSIM: {('-' if lm.get('ssim') in (None,-1) else f'{lm.get('ssim'):.4f}')} / 差分率: {lm.get('diff_ratio','-') if isinstance(lm.get('diff_ratio'),str) else f'{(lm.get('diff_ratio') or 0)*100:.2f}%'} / ボックス数: {lm.get('boxes','-')}")

    # 履歴ギャラリー
    st.markdown("---")
    st.subheader("📚 履歴ギャラリー")
    caps = sorted(glob.glob(str(Path(ss.session_dir)/"captures/*.jpg")))
    ncols = st.slider("列数", 2, 6, 4, 1, key="k_cols")
    cols = st.columns(ncols)
    for i, cp in enumerate(caps):
        with cols[i % ncols]:
            ts = Path(cp).stem
            dp = Path(ss.session_dir)/"diffs"/f"{ts}_diff.png"
            st.image(bgr2rgb(cv2.imread(cp)), caption=ts, use_container_width=True)
            if dp.exists():
                with st.expander("差分を表示"):
                    st.image(bgr2rgb(cv2.imread(str(dp))), use_container_width=True)

# ============ ルーティング（フェーズ遷移） ============

def main():
    # ナビ（上部右寄せ）
    st.caption(f"Auto-refresh: {_AUTOREFRESH_IMPL}  •  セッション: `{ss.session_dir or '未作成'}`")

    phase = ss.get("phase","INIT")
    if phase == "INIT":
        ui_step_baseline()
    elif phase == "BASELINE_SET":
        st.success("基準画像 OK！つぎは設定です。")
        ui_step_config()
    elif phase == "CONFIG_SET":
        st.info("設定が保存されました。スタートできます！")
        ui_step_run()
    elif phase in ("RUNNING", "PAUSED"):
        ui_step_run()
    else:
        st.warning("未知の状態です。初期化します。")
        _init_state()
        ui_step_baseline()

if __name__ == "__main__":
    main()