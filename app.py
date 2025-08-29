# app.py
# -*- coding: utf-8 -*-
"""
📷 基準撮影 → 設定 → 監視 の3ステップUI + YOLO物体検出
- 起動直後は「基準を撮影/読み込み」のみ表示
- 基準確定後に設定（撮影間隔・整列など）
- スタートでループ（基準 vs 現在）差分をリアルタイム表示
- すべての撮影＆差分は runs/<session>/ に保存、履歴ギャラリーで見返し＆ZIP DL
- YOLO: 基準時に検出した物体一覧を保存。監視中に「新規出現/消失」を検知して通知
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
from typing import Tuple, Dict, Any, Optional, List

# --- Optional: click-to-pick (fallback to sliders if unavailable)
try:
    from st_click_detector import click_detector as _click_detector  # pip install st-click-detector
    _HAS_CLICK = True
except Exception:
    _HAS_CLICK = False

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
from core.vis import make_boxes_overlay_transparent, draw_clusters_only
from core.cluster import cluster_dense_boxes
from core.diff import difference_boxes, fused_diff_mask, boxes_from_mask
from core.quality import quality_harmonize_pair

# ---- YOLO ----
#   pip install ultralytics
#   別ファイル core/yolo.py に detect_objects / draw_detections を実装済みの想定
from core.yolo import detect_objects, draw_detections

# ============ 色ターゲット監視（Labベース） ヘルパ ============

def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def _clip(v, lo, hi):  # small util
    return max(lo, min(int(v), hi))


def _scope_roi(img_shape: tuple, anchor_xy: Optional[tuple], scope_pct: float) -> Optional[tuple]:
    """
    監視範囲を%で指定してROIを自動生成。
    - scope_pct >= 99.5 → 画像全体(Noneを返す)
    - それ未満 → アンカー(x,y)中心の正方ROI（面積が画像の scope_pct % 目安）
    """
    h, w = img_shape[:2]
    if scope_pct is None:
        scope_pct = 100.0
    if scope_pct >= 99.5 or not anchor_xy:
        return None
    # 面積比から一辺を計算（正方形）。最小サイズは4px。
    side = int(max(4, round((scope_pct / 100.0) ** 0.5 * min(w, h))))
    cx, cy = int(anchor_xy[0]), int(anchor_xy[1])
    x = _clip(cx - side // 2, 0, max(0, w - side))
    y = _clip(cy - side // 2, 0, max(0, h - side))
    return (x, y, side, side)

def draw_target_preview(img: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    vis = img.copy()
    x = int(x); y = int(y); r = int(r)
    cv2.circle(vis, (x, y), r, (0, 255, 255), 2)
    cv2.drawMarker(vis, (x, y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    return vis

def clickable_pick_xy(img_rgb: np.ndarray, key: str = "clickpick") -> tuple[int | None, int | None]:
    if not _HAS_CLICK:
        st.image(img_rgb, use_container_width=True)
        return None, None
    h, w = img_rgb.shape[:2]
    events = _click_detector(img_rgb, key=key)
    if events and "x" in events and "y" in events and events["x"] is not None and events["y"] is not None:
        disp_w = events.get("display_width", w) or w
        disp_h = events.get("display_height", h) or h
        rx = float(events["x"]) / float(max(1, disp_w))
        ry = float(events["y"]) / float(max(1, disp_h))
        x = int(np.clip(round(rx * w), 0, w - 1))
        y = int(np.clip(round(ry * h), 0, h - 1))
        return x, y
    return None, None

def sample_lab_stats(img: np.ndarray, x: int, y: int, r: int, use_l: bool = False) -> dict:
    h, w = img.shape[:2]
    x = int(np.clip(x, 0, w - 1)); y = int(np.clip(y, 0, h - 1)); r = int(max(1, r))
    lab = _lab(img)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    if mask.sum() < 5:
        r = max(3, r + 2)
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    vals = lab[mask]
    mean = vals.mean(axis=0)
    std = vals.std(axis=0) + 1e-6
    return {"mean": (float(mean[0]), float(mean[1]), float(mean[2])),
            "std": (float(std[0]), float(std[1]), float(std[2])), "use_l": bool(use_l)}

def _target_mask_for_img(img: np.ndarray, target: dict, roi: Optional[tuple] = None) -> np.ndarray:
    lab = _lab(img)
    h, w = lab.shape[:2]
    if roi is not None:
        x, y, rw, rh = [int(v) for v in roi]
        x = max(0, x); y = max(0, y); rw = max(1, rw); rh = max(1, rh)
        x2 = min(w, x + rw); y2 = min(h, y + rh)
        sub = lab[y:y2, x:x2]; off = (x, y)
    else:
        sub = lab; off = (0, 0)
    meanL, meana, meanb = target.get("mean", (0, 0, 0))
    stdL, stda, stdb = target.get("std", (1, 1, 1))
    k = float(target.get("k_sigma", 2.5))
    use_l = bool(target.get("use_l", False))
    L = sub[..., 0]; A = sub[..., 1]; B = sub[..., 2]
    da = (A - meana) / (stda + 1e-6)
    db = (B - meanb) / (stdb + 1e-6)
    dist2 = da * da + db * db
    if use_l:
        dL = (L - meanL) / (stdL + 1e-6)
        dist2 = dist2 + dL * dL
    mask = (dist2 <= (k * k)).astype(np.uint8) * 255
    out = np.zeros((h, w), np.uint8)
    out[off[1]:off[1] + mask.shape[0], off[0]:off[0] + mask.shape[1]] = mask
    return out

def ratio_for_target(img: np.ndarray, target: dict) -> float:
    """
    ターゲット色の占有率（0..1）
    - target["scope_pct"] と target["anchor"] があればそれに基づくROIを自動生成
    - 明示的 target["roi"] があればそれを優先
    """
    explicit_roi = target.get("roi")
    scope_pct = float(target.get("scope_pct", 100.0))
    anchor = target.get("anchor")  # (x,y)
    roi = explicit_roi if explicit_roi is not None else _scope_roi(img.shape, anchor, scope_pct)
    m = _target_mask_for_img(img, target, roi)
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        denom = max(1, w * h)
        return float((m[y:y+h, x:x+w] > 0).sum()) / float(denom)
    else:
        return float((m > 0).sum()) / float(m.size)

def render_color_target_editor():
    if not ss.get("baseline_path"):
        st.info("基準画像が未設定です。"); return
    base = cv2.imread(ss.baseline_path, cv2.IMREAD_COLOR)
    if base is None:
        st.warning("基準画像の読み込みに失敗しました。"); return
    h, w = base.shape[:2]
    cols = st.columns([2,1])
    with cols[0]:
        st.caption("基準画像上で色ターゲットを指定します。クリック非対応ならスライダーを使用。")
        img_rgb = bgr2rgb(base)
        cx, cy = clickable_pick_xy(img_rgb, key="pick_target_xy_cfg")
        default_x = w//2 if cx is None else cx
        default_y = h//2 if cy is None else cy
        x = st.slider("X", 0, max(1, w-1), int(default_x), key="t_x")
        y = st.slider("Y", 0, max(1, h-1), int(default_y), key="t_y")
        r = st.slider("半径 r", 3, max(5, min(h,w)//4), 20, key="t_r")
        preview = draw_target_preview(img_rgb, x, y, r)
        st.image(preview, caption="位置プレビュー", use_container_width=True)
        scope_pct = st.slider("監視範囲（%）", 1, 100, 100, help="100%: 画面全体（移動に寛容） / 1%: アンカー周辺のみ（移動に敏感）", key="t_scope")
    with cols[1]:
        name = st.text_input("名前", value=f"target_{len(ss.targets)+1}", key="t_name")
        direction = st.selectbox("アラート条件", ["decrease", "increase"], help="decrease: 減ったら警告 / increase: 増えたら警告", key="t_dir")
        threshold_pct = st.slider("変化しきい値(%)", 1, 50, 10, key="t_thr")
        # ▼ ヘルプ（色ターゲット：変化しきい値）
        _help_expander(*_HELP_MD["color_threshold"])
        k_sigma = st.slider("色ゆるさ kσ", 1.0, 4.0, 2.5, 0.1, key="t_ks")
        # ▼ ヘルプ（色ターゲット：kσ）
        _help_expander(*_HELP_MD["color_k_sigma"])
        use_l = st.checkbox("明度Lも使う", value=False, help="色味メインで十分ならOFF推奨", key="t_useL")
        roi_enable = st.checkbox("高度: ROIを手動指定（任意）", value=False, key="t_roi_en")
        roi = None
        if roi_enable:
            rx = st.slider("ROI x", 0, w-1, 0, key="t_rx")
            ry = st.slider("ROI y", 0, h-1, 0, key="t_ry")
            rw = st.slider("ROI w", 1, w-rx, w, key="t_rw")
            rh = st.slider("ROI h", 1, h-ry, h, key="t_rh")
            roi = (rx, ry, rw, rh)
        if st.button("＋ この色をターゲットに追加", type="primary", use_container_width=True, key="t_add"):
            stats = sample_lab_stats(base, x, y, r, use_l=use_l)
            tmp_target = {
                "name": name,
                "mean": stats["mean"], "std": stats["std"], "use_l": use_l,
                "k_sigma": float(k_sigma),
                "direction": direction,
                "threshold_pct": float(threshold_pct),
                # 位置アンカーと監視範囲%（100%なら全体）
                "anchor": (int(x), int(y)),
                "scope_pct": float(scope_pct),
                # 明示ROIがある場合はそれを優先
                "roi": roi
            }
            base_ratio = ratio_for_target(base, tmp_target)
            tmp_target["base_ratio"] = float(base_ratio)
            ss.targets.append(tmp_target)
            st.success(f"追加しました: {name} (基準割合 {base_ratio*100:.2f}%)")
        show_mask_preview = st.checkbox("この設定でマスクをプレビュー", value=False, key="t_preview")
        if show_mask_preview:
            tmp = {
                "name": name, "k_sigma": float(k_sigma), "use_l": use_l,
                "mean": sample_lab_stats(base, x, y, r, use_l)["mean"],
                "std": sample_lab_stats(base, x, y, r, use_l)["std"],
                "anchor": (int(x), int(y)),
                "scope_pct": float(scope_pct),
                "roi": roi
            }
            roi_preview = roi if roi is not None else _scope_roi(base.shape, tmp.get("anchor"), tmp.get("scope_pct", 100.0))
            pm = _target_mask_for_img(base, tmp, roi_preview)
            st.image(pm, clamp=True, caption="ターゲットマスク（基準画像）", use_container_width=True)
    if ss.targets:
        st.markdown("#### 追加済みターゲット")
        del_idxs = []
        for i, t in enumerate(ss.targets):
            cols2 = st.columns([3,2,2,2,1])
            cols2[0].write(f"**{t['name']}**")
            cols2[1].write(f"条件: {t['direction']} / しきい値 {t['threshold_pct']}%")
            cols2[2].write(f"kσ={t.get('k_sigma',2.5)} / L使用={t.get('use_l',False)}")
            cols2[3].write(f"基準割合: {t.get('base_ratio',0.0)*100:.2f}% / 監視範囲: {int(t.get('scope_pct',100))}%")
            if cols2[4].button("🗑️", key=f"cfg_del_t_{i}"):
                del_idxs.append(i)
        for i in reversed(del_idxs):
            ss.targets.pop(i)

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

# ---- モード切替（監視 / テスト）----
ss = st.session_state
ss.setdefault("mode", "監視")
ss.setdefault("test_pairs", [])  # [(pathA, pathB)]
mode = st.sidebar.radio("モード", ["監視", "テスト"], index=(0 if ss.get("mode","監視")=="監視" else 1))
ss.mode = mode

# ============ セッション状態 ============

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
    ss.setdefault("targets", [])
    # YOLO関連
    ss.setdefault("baseline_yolo", [])      # [{"label","conf","bbox"}]
    ss.setdefault("yolo_conf", 0.5)
    ss.setdefault("yolo_watch_person", True)
    ss.setdefault("yolo_alert_new", True)
    ss.setdefault("yolo_alert_missing", True)
    ss.setdefault("last_target_results", [])

_init_state()

# ============ ラン（保存先） ============

def new_session_dir() -> str:
    root = Path("runs"); root.mkdir(exist_ok=True)
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
    if out.exists(): out.unlink()
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

# ===== ヘルプ（expander方式）共通UI =====
def _help_expander(title: str, body_md: str, default_open: bool = False):
    """ラベル直下に展開できる説明ブロックを出す。"""
    with st.expander("❓ " + title, expanded=default_open):
        st.markdown(body_md)

_HELP_MD = {
    "min_wh": (
        "最小ボックス幅/高さ（px）＝ min_wh",
        """
- 差分から作る小ボックスの最小サイズ。これ未満はノイズとして捨てる。
- 大きくすると: ノイズ減るが細かい変化を見落とす。
- 小さくすると: 細かい変化も拾うがノイズも増える。
- 目安: **15 px**。
"""
    ),
    "yolo_conf": (
        "YOLO信頼度しきい値 ＝ yolo_conf",
        """
- YOLOv8 の検出信頼度の下限。
- 高くすると: 誤検出減るが見落とし増える。
- 低くすると: 見落とし減るが誤検出増える。
- 目安: **0.5**。
"""
    ),
    "target_short": (
        "共通ダウンサンプル短辺（px）＝ target_short",
        """
- 基準と現在フレームの短辺をこのピクセル数に揃えてから全処理を行う（解像度統一）。
- 大きくすると: 細部まで見えるが重くなり、ノイズも拾いやすい。
- 小さくすると: 軽くなるが細かい変化は消えやすい。
- 目安: **720 前後**。
"""
    ),
    "shift_px_thresh": (
        "ズレ判定しきい値（px）＝ shift_px_thresh",
        """
- カメラ位置のズレ量がこのピクセルを超えると「ズレあり」と判定 → 自動整列を試みる。
- 小さくすると: わずかなズレでも整列（敏感）。
- 大きくすると: 軽微なズレは無視（安定）。
- 目安: **6 px**。
"""
    ),
    "color_threshold": (
        "変化しきい値(%)（色ターゲット）",
        """
- ターゲット色の画素比が基準からどれくらい増減したらアラートを出すか。
- 大きくすると: 鈍感（大きな変化でのみ警告）。
- 小さくすると: 敏感（小さな変化ですぐ警告）。
- 目安: **10〜20%**。
"""
    ),
    "color_k_sigma": (
        "色ゆるさ kσ（色ターゲット）",
        """
- Lab色空間での「基準色の平均±k×標準偏差」を色領域として採用。
- 大きくすると: 色のブレに強いが似た色も拾いやすい。
- 小さくすると: 厳密に色を見られるが少しの変化で見失いやすい。
- 目安: **2.0〜2.5**。
"""
    ),
}

# ======= Beep/Alert Helpers =======
def _play_beep(duration_ms: int = 500, freq_hz: int = 880):
    st.components.v1.html(f"""
    <script>
      (function(){{
        try {{
          const AC = window.AudioContext || window.webkitAudioContext;
          const ctx = new AC();
          const o = ctx.createOscillator();
          const g = ctx.createGain();
          o.type = "sine";
          o.frequency.value = {freq_hz};
          o.connect(g); g.connect(ctx.destination);
          g.gain.setValueAtTime(0.001, ctx.currentTime);
          g.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.02);
          o.start();
          g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + {duration_ms}/1000.0);
          o.stop(ctx.currentTime + {duration_ms}/1000.0 + 0.05);
        }} catch(e) {{ console.log("beep error", e); }}
      }})();
    </script>
    """, height=0)

def _alert_banner(msg: str):
    st.markdown(
        "<div style='padding:12px 16px;border-radius:10px;border:2px solid #ffb3b3;background:#fff1f1;color:#b00020;font-weight:700;font-size:20px;'>"
        f"🚨 {msg}"
        "</div>",
        unsafe_allow_html=True
    )
    _play_beep(380, 920); _play_beep(380, 720)

# 画像読込ヘルパー
def imread_color(path: str) -> Optional[np.ndarray]:
    if not path: return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img if isinstance(img, np.ndarray) and img.size > 0 else None

# ============ YOLO 比較ユーティリティ ============

def _iou_xywh(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah; bx2, by2 = bx+bw, by+bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = aw*ah + bw*bh - inter
    return float(inter) / float(max(1, union))

def analyze_yolo_changes(baseline_det: List[Dict[str,Any]], current_det: List[Dict[str,Any]], iou_thr: float=0.3):
    """
    基準と現在のYOLO検出を突き合わせて
    - new_labels: 基準に無いラベルが現れた
    - missing_labels: 基準にあったラベルが消えた
    - person_appeared: 基準にperson無し→現在person有り

    入力の dict 形式に差があっても耐えるよう、以下のキーに対応：
      ラベル: "label" | "name" | "class_name" | "cls_name" | "cls" | "class"
      BBox :  "bbox"(xywh) | "xywh" | "xyxy" | {"x","y","w","h"}
    """
    def _label_of(d: Dict[str,Any]) -> Optional[str]:
        for k in ("label","name","class_name","cls_name","cls","class"):
            if k in d and d[k] is not None:
                return str(d[k])
        return None

    def _bbox_of(d: Dict[str,Any]) -> Optional[tuple]:
        if "bbox" in d and d["bbox"] is not None:
            x, y, w, h = d["bbox"]
            return (float(x), float(y), float(w), float(h))
        if "xywh" in d and d["xywh"] is not None:
            x, y, w, h = d["xywh"]
            return (float(x), float(y), float(w), float(h))
        if "xyxy" in d and d["xyxy"] is not None:
            x1, y1, x2, y2 = d["xyxy"]
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            return (float(x1), float(y1), w, h)
        if isinstance(d.get("box"), dict):
            bx = d["box"]
            if all(k in bx for k in ("x","y","w","h")):
                return (float(bx["x"]), float(bx["y"]), float(bx["w"]), float(bx["h"]))
        return None

    base_labels = [lb for lb in (_label_of(d) for d in baseline_det) if lb]
    cur_labels  = [lb for lb in (_label_of(d) for d in current_det) if lb]

    new_labels = sorted(list(set(cur_labels) - set(base_labels)))
    missing_labels = sorted(list(set(base_labels) - set(cur_labels)))

    def _matched(s: Dict[str,Any], t: Dict[str,Any]) -> bool:
        slb, tlb = _label_of(s), _label_of(t)
        if not slb or not tlb or slb != tlb:
            return False
        sbx, tbx = _bbox_of(s), _bbox_of(t)
        if not sbx or not tbx:
            return False
        return _iou_xywh(sbx, tbx) >= iou_thr

    missing_instances = [s for s in baseline_det if not any(_matched(s, t) for t in current_det)]
    appeared_instances = [t for t in current_det if not any(_matched(s, t) for s in baseline_det)]

    person_appeared = ("person" not in base_labels) and ("person" in cur_labels)
    person_missing  = ("person" in base_labels) and ("person" not in cur_labels)

    return {
        "new_labels": new_labels,
        "missing_labels": missing_labels,
        "appeared_instances": appeared_instances,
        "missing_instances": missing_instances,
        "person_appeared": person_appeared,
        "person_missing": person_missing
    }
# ============ 差分処理（基準 vs 現在） ============

def compare_to_baseline(baseline_bgr: np.ndarray, current_bgr: np.ndarray, cfg: Dict[str,Any]) -> Dict[str,Any]:
    # 画質整合（同倍率ダウンサンプル・シャープ/明るさ合わせ）
    a, b = quality_harmonize_pair(baseline_bgr, current_bgr,
                                  target_short=cfg.get("target_short", 720),
                                  match_sharp=True, match_hist=True, denoise='none')
    a, b = ensure_same_size(a, b)

    # ▼ 色ターゲット用：ヒストグラム補正なし版（色ずれに強くする）
    a_color, b_color = quality_harmonize_pair(baseline_bgr, current_bgr,
                                              target_short=cfg.get("target_short", 720),
                                              match_sharp=True, match_hist=False, denoise='none')
    a_color, b_color = ensure_same_size(a_color, b_color)

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
    if aligned_method == "NONE":
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
        ss_val = ssim(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    except Exception:
        ss_val = None

    # 色ターゲット監視
    targets_cfg = cfg.get("targets", []) or []
    target_results = []
    for t in targets_cfg:
        try:
            curr_ratio = ratio_for_target(b_color, t)
            base_ratio = float(t.get("base_ratio", 0.0))
            direction = t.get("direction", "decrease")
            thr = float(t.get("threshold_pct", 5.0)) / 100.0
            delta = curr_ratio - base_ratio
            if direction == "increase":
                alert = (delta >= thr); vanished = False
            else:
                alert = (-delta >= thr)
                # 完全消失の下限を基準比に応じて可変に（誤爆低減）
                # 例: base_ratio*0.2 を下限としつつ、絶対下限は 0.001 (=0.1%) より小さくしない
                vanish_floor = max(0.001, base_ratio * 0.2)
                vanished = (curr_ratio <= vanish_floor)
            target_results.append({
                "name": t.get("name", "target"),
                "curr_ratio": curr_ratio,
                "base_ratio": base_ratio,
                "delta": delta,
                "direction": direction,
                "threshold_pct": t.get("threshold_pct", 5.0),
                "alert": bool(alert),
                "vanished": bool(vanished)
            })
        except Exception as e:
            target_results.append({"name": t.get("name","target"), "error": str(e)})

    # YOLO物体検出（現在フレーム）
    yolo_conf = float(cfg.get("yolo_conf", ss.get("yolo_conf", 0.5)))
    det_cur = detect_objects(b, conf=yolo_conf)
    yolo_vis = draw_detections(b, det_cur)

    # YOLO 差分（基準 vs 現在）
    det_base = ss.get("baseline_yolo", [])
    yolo_changes = analyze_yolo_changes(det_base, det_cur, iou_thr=0.3)

    return {
        "A": a, "B": b,
        "vis_boxes": vis_boxes, "mask": th_mask, "overlay": overlay, "clusters_vis": clusters_vis,
        "boxes": boxes, "diff_ratio": diff_ratio, "ssim": ss_val,
        "aligned": aligned_method,
        "target_results": target_results, "target_overlay": a.copy(),  # overlay of targets is optional; keep a for layout
        "yolo_detections": det_cur, "yolo_vis": yolo_vis, "yolo_changes": yolo_changes
    }

# ============ テスト実行ユーティリティ ============
def run_test_case(pathA: str, pathB: str, cfg: Dict[str,Any], out_dir: Path) -> Dict[str,Any]:
    a = imread_color(pathA); b = imread_color(pathB)
    if a is None or b is None:
        return {"ok": False, "error": f"読込失敗 A={pathA}, B={pathB}"}
    res = compare_to_baseline(a, b, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir/"A_baseline.png"), a)
    cv2.imwrite(str(out_dir/"B_compare.png"), b)
    cv2.imwrite(str(out_dir/"diff_boxes.png"), res["vis_boxes"])
    cv2.imwrite(str(out_dir/"diff_mask.png"), res["mask"])
    cv2.imwrite(str(out_dir/"diff_boxesx2.png"), res["clusters_vis"])
    # YOLO可視化
    if isinstance(res.get("yolo_vis"), np.ndarray):
        cv2.imwrite(str(out_dir/"yolo_vis.png"), res["yolo_vis"])
    return {
        "ok": True, "A": pathA, "B": pathB,
        "aligned": res["aligned"], "ssim": res["ssim"],
        "diff_ratio": res["diff_ratio"], "boxes": len(res["boxes"]),
        "dir": str(out_dir)
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
                # --- YOLO: 基準検出を保存 ---
                yolo_conf = float(ss.get("yolo_conf", 0.5))
                det_base = detect_objects(frame, conf=yolo_conf)
                ss.baseline_yolo = det_base
                with open(Path(session)/"baseline_yolo.json","w",encoding="utf-8") as f:
                    json.dump(det_base, f, ensure_ascii=False, indent=2)
                # 可視化も保存
                vis = draw_detections(frame, det_base)
                cv2.imwrite(str(Path(session)/"baseline_yolo_vis.png"), vis)

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
                # YOLO基準
                yolo_conf = float(ss.get("yolo_conf", 0.5))
                det_base = detect_objects(img, conf=yolo_conf)
                ss.baseline_yolo = det_base
                with open(Path(session)/"baseline_yolo.json","w",encoding="utf-8") as f:
                    json.dump(det_base, f, ensure_ascii=False, indent=2)
                vis = draw_detections(img, det_base)
                cv2.imwrite(str(Path(session)/"baseline_yolo_vis.png"), vis)

                ss.phase = "BASELINE_SET"
                st.success("基準を保存しました！次へ進めます。")
                st.rerun()

    # 基準ができたら色ターゲット追加UI + YOLO基準表示
    if ss.get("baseline_path"):
        st.markdown("---")
        st.subheader("🎯 監視する色ターゲットを追加（任意）")
        render_color_target_editor()

        with st.expander("🧠 基準のYOLO検出（参考）", expanded=False):
            st.caption(f"しきい値 conf ≥ {ss.get('yolo_conf',0.5)}")
            if Path(ss.session_dir, "baseline_yolo_vis.png").exists():
                st.image(bgr2rgb(cv2.imread(str(Path(ss.session_dir, "baseline_yolo_vis.png")))), use_container_width=True)
            st.json(ss.get("baseline_yolo", []))

# ============ UI：ステップ 2（設定） ============

def ui_step_config():
    st.header("② 設定")
    with st.form("cfg_form", clear_on_submit=False):
        st.markdown("**撮影**")
        interval = st.slider("撮影間隔（秒）⏱️", 10, 600, 60, 5)
        target_short = st.slider("共通ダウンサンプル短辺（px）", 360, 1440, 720, 60)
        # ▼ ヘルプ（target_short）
        _help_expander(*_HELP_MD["target_short"])

        st.markdown("---")
        st.markdown("**整列と検出**")
        align_mode = st.selectbox("整列モード", ["AUTO","ECC","H","OFF"], help="AUTO推奨")
        shift_thresh = st.slider("ズレ判定しきい値(px)", 1.0, 20.0, 6.0, 0.5)
        # ▼ ヘルプ（shift_px_thresh）
        _help_expander(*_HELP_MD["shift_px_thresh"])
        min_wh = st.slider("最小ボックス幅/高さ(px)", 5, 100, 15, 1)
        # ▼ ヘルプ（min_wh）
        _help_expander(*_HELP_MD["min_wh"])

        st.markdown("---")
        st.markdown("**YOLO（物体監視）**")
        yolo_conf = st.slider("YOLO信頼度しきい値", 0.1, 0.9, float(ss.get("yolo_conf",0.5)), 0.05)
        # ▼ ヘルプ（yolo_conf）
        _help_expander(*_HELP_MD["yolo_conf"])
        watch_person = st.toggle("人の出現を監視（基準に居なかったのに現れたら通知）", value=bool(ss.get("yolo_watch_person", True)))
        alert_new = st.toggle("基準にないラベルが現れたら通知", value=bool(ss.get("yolo_alert_new", True)))
        alert_missing = st.toggle("基準にあったラベルが消えたら通知", value=bool(ss.get("yolo_alert_missing", True)))

        st.markdown("---")
        submitted = st.form_submit_button("✅ 設定を保存して次へ", use_container_width=True)
        if submitted:
            ss.yolo_conf = float(yolo_conf)
            ss.yolo_watch_person = bool(watch_person)
            ss.yolo_alert_new = bool(alert_new)
            ss.yolo_alert_missing = bool(alert_missing)
            ss.config = {
                "interval": int(interval),
                "target_short": int(target_short),
                "align_mode": align_mode,
                "shift_px_thresh": float(shift_thresh),
                "min_wh": int(min_wh),
                "camera": {"index": ss.cam_index, "backend": ss.backend},
                "targets": ss.get("targets", []),
                # YOLO
                "yolo_conf": float(yolo_conf),
                "yolo_watch_person": bool(watch_person),
                "yolo_alert_new": bool(alert_new),
                "yolo_alert_missing": bool(alert_missing),
            }
            save_config(ss.session_dir, ss.config)
            ss.phase = "CONFIG_SET"
            ss.next_shot_ts = 0.0
            st.success("設定を保存しました！")
            st.rerun()

    st.markdown("---")
    with st.expander("🎯 監視する色ターゲットを設定（任意）", expanded=True):
        render_color_target_editor()

# ============ UI：ステップ 3（監視） ============

def ui_step_run():
    st.header("③ 監視スタート")
    colA, colB = st.columns([2,1])

    with colB:
        running = ss.get("phase") == "RUNNING"
        status_badge = f"<span class='big-badge {'ok-badge' if running else 'ng-badge'}'>{'🟢 監視中' if running else '🔴 停止中'}</span>"
        st.markdown(status_badge, unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        if c1.button("▶️ スタート", use_container_width=True):
            ss.phase = "RUNNING"
            ss.next_shot_ts = time.time() + ss.config["interval"]
            st.rerun()
        if c2.button("⏸ 停止", use_container_width=True):
            ss.phase = "PAUSED"
            st.rerun()

        st.markdown("---")
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
            tick(); setInterval(tick, 250);
            </script>
            """, height=60)
        else:
            st.caption("停止中（カウントダウンなし）")

        if st.button("💾 このセッションをZIPでダウンロード", use_container_width=True):
            zpath = make_zip(ss.session_dir)
            with open(zpath, "rb") as f:
                st.download_button("ZIPをダウンロード", f, file_name=Path(zpath).name, mime="application/zip", use_container_width=True)

        st.markdown("---")
        st.caption("基準画像プレビュー / 基準YOLO")
        base = cv2.imread(ss.baseline_path, cv2.IMREAD_COLOR)
        if base is not None:
            st.image(bgr2rgb(base), use_container_width=True)
        if Path(ss.session_dir, "baseline_yolo_vis.png").exists():
            st.image(bgr2rgb(cv2.imread(str(Path(ss.session_dir, "baseline_yolo_vis.png")))), caption="基準YOLO", use_container_width=True)

    with colA:
        st.subheader("リアルタイム差分")
        ph_cols = st.columns(3)
        phA, phB, phD = ph_cols[0].empty(), ph_cols[1].empty(), ph_cols[2].empty()
        info = st.empty()

        running = ss.get("phase") == "RUNNING"
        if running:
            st_autorefresh(interval=max(1000, int(ss.config.get("interval", 60) * 1000)), key="tick-live")

        if ss.phase == "RUNNING" and time.time() >= ss.next_shot_ts:
            frame, diag = capture_frame(ss.config["camera"]["index"], ss.config["camera"]["backend"])
            if frame is not None:
                baseline = cv2.imread(ss.baseline_path, cv2.IMREAD_COLOR)
                res = compare_to_baseline(baseline, frame, ss.config)

                tsname = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                cap_path = str(Path(ss.session_dir)/"captures"/f"{tsname}.jpg")
                diff_path = str(Path(ss.session_dir)/"diffs"/f"{tsname}_diff.png")
                diffx2_path = str(Path(ss.session_dir)/"diffs"/f"{tsname}_diffx2.png")
                mask_path = str(Path(ss.session_dir)/"diffs"/f"{tsname}_mask.png")
                yolo_path = str(Path(ss.session_dir)/"diffs"/f"{tsname}_yolo.png")
                cv2.imwrite(cap_path, frame)
                cv2.imwrite(diff_path, res["vis_boxes"])
                cv2.imwrite(diffx2_path, res["clusters_vis"])
                cv2.imwrite(mask_path, res["mask"])
                if isinstance(res.get("yolo_vis"), np.ndarray):
                    cv2.imwrite(yolo_path, res["yolo_vis"])

                append_index(ss.session_dir, {
                    "ts": tsname, "capture": Path(cap_path).name,
                    "aligned": res["aligned"], "ssim": (res["ssim"] if res["ssim"] is not None else -1),
                    "diff_ratio": res["diff_ratio"], "boxes": len(res["boxes"])
                })
                ss.last_metrics = {"ts": tsname, "aligned": res["aligned"], "ssim": res["ssim"], "diff_ratio": res["diff_ratio"], "boxes": len(res["boxes"])}
                ss.last_target_results = res.get("target_results", [])

                # ---- YOLOアラート ----
                yc = res.get("yolo_changes", {})
                msgs = []
                if ss.config.get("yolo_watch_person", True) and yc.get("person_appeared"):
                    msgs.append("人が現れました！")
                if ss.config.get("yolo_alert_new", True) and yc.get("new_labels"):
                    msgs.append("新しい物体: " + ", ".join(yc["new_labels"]))
                if ss.config.get("yolo_alert_missing", True) and yc.get("missing_labels"):
                    msgs.append("消えた物体: " + ", ".join(yc["missing_labels"]))
                # ---- 色ターゲットのアラート ----
                tr = res.get("target_results", []) or []
                for t in tr:
                    if t.get("alert"):
                        direction_jp = "増加" if t.get("direction") == "increase" else "減少"
                        msgs.append(
                            f"色[{t.get('name','target')}] {direction_jp} "
                            f"{abs(t.get('delta',0)*100):.1f}% "
                            f"(基準 {t.get('base_ratio',0)*100:.1f}% → 現在 {t.get('curr_ratio',0)*100:.1f}%)"
                        )
                    if t.get("vanished"):
                        msgs.append(f"色[{t.get('name','target')}] 完全に消失しました")
                if msgs:
                    _alert_banner(" / ".join(msgs))

            ss.next_shot_ts = time.time() + ss.config["interval"]

        # 表示（最新）
        baseline = cv2.imread(ss.baseline_path, cv2.IMREAD_COLOR)
        try:
            latest_list = sorted(glob.glob(str(Path(ss.session_dir)/"captures/*.jpg")))
            latest_cap = latest_list[-1] if latest_list else None
        except Exception:
            latest_cap = None
        latest_diff = latest_diffx2 = latest_mask = latest_yolo = None
        if latest_cap:
            tsname = Path(latest_cap).stem
            latest_diff = Path(ss.session_dir)/"diffs"/f"{tsname}_diff.png"
            latest_diffx2 = Path(ss.session_dir)/"diffs"/f"{tsname}_diffx2.png"
            latest_mask = Path(ss.session_dir)/"diffs"/f"{tsname}_mask.png"
            latest_yolo = Path(ss.session_dir)/"diffs"/f"{tsname}_yolo.png"

        if latest_cap and baseline is not None:
            phA.image(bgr2rgb(baseline), caption="基準", use_container_width=True)
            imgB = cv2.imread(latest_cap, cv2.IMREAD_COLOR)
            phB.image(bgr2rgb(imgB), caption="今回ショット", use_container_width=True)
            grid = st.columns([2,2,2])
            with grid[0]:
                if latest_diff and latest_diff.exists():
                    phD.image(bgr2rgb(cv2.imread(str(latest_diff))), caption="差分（赤枠・小）", use_container_width=True)
            with grid[1]:
                if latest_diffx2 and latest_diffx2.exists():
                    st.image(bgr2rgb(cv2.imread(str(latest_diffx2))), caption="差分（大枠クラスタ）", use_container_width=True)
            with grid[2]:
                if latest_mask and latest_mask.exists():
                    st.image(cv2.imread(str(latest_mask), cv2.IMREAD_GRAYSCALE), clamp=True, caption="差分マスク", use_container_width=True)

            if latest_yolo and latest_yolo.exists():
                st.image(bgr2rgb(cv2.imread(str(latest_yolo))), caption="YOLO物体検出", use_container_width=True)

            # 色ターゲット結果の可視化（直近ショットの結果を常時表示）
            with st.expander("🎯 色ターゲットの結果", expanded=False):
                tr = ss.get("last_target_results", [])
                if tr:
                    try:
                        import pandas as pd
                        rows = []
                        for t in tr:
                            if "error" in t:
                                rows.append({
                                    "name": t.get("name","target"),
                                    "error": t["error"]
                                })
                            else:
                                rows.append({
                                    "name": t.get("name","target"),
                                    "dir": t.get("direction"),
                                    "base(%)": round(t.get("base_ratio",0)*100, 2),
                                    "curr(%)": round(t.get("curr_ratio",0)*100, 2),
                                    "delta(%)": round(t.get("delta",0)*100, 2),
                                    "alert": bool(t.get("alert", False)),
                                    "vanished": bool(t.get("vanished", False)),
                                })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    except Exception:
                        for t in tr:
                            if "error" in t:
                                st.error(f"[{t.get('name','target')}] error: {t['error']}")
                            else:
                                st.write(
                                    f"- **{t.get('name','target')}** "
                                    f"(基準 {t.get('base_ratio',0)*100:.2f}% → 現在 {t.get('curr_ratio',0)*100:.2f}%, "
                                    f"Δ {t.get('delta',0)*100:.2f}%, dir={t.get('direction')}) "
                                    f"{'⚠️ alert' if t.get('alert') else ''} "
                                    f"{'❌ vanished' if t.get('vanished') else ''}"
                                )
                else:
                    st.caption("直近ショットの結果がまだありません。（①や②でターゲット追加→③でスタートしてください）")

        lm = ss.get("last_metrics") or {}
        info.info(
            f"整列: {lm.get('aligned','-')} / "
            f"SSIM: {('-' if lm.get('ssim') in (None,-1) else f'{lm.get('ssim'):.4f}')} / "
            f"差分率: {lm.get('diff_ratio','-') if isinstance(lm.get('diff_ratio'),str) else f'{(lm.get('diff_ratio') or 0)*100:.2f}%'} / "
            f"ボックス数: {lm.get('boxes','-')}"
        )

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
            dpx2 = Path(ss.session_dir)/"diffs"/f"{ts}_diffx2.png"
            yp = Path(ss.session_dir)/"diffs"/f"{ts}_yolo.png"
            st.image(bgr2rgb(cv2.imread(cp)), caption=ts, use_container_width=True)
            with st.expander("差分/YOLOを表示"):
                if dp.exists():
                    st.image(bgr2rgb(cv2.imread(str(dp))), caption="差分（赤枠・小）", use_container_width=True)
                if dpx2.exists():
                    st.image(bgr2rgb(cv2.imread(str(dpx2))), caption="差分（大枠クラスタ）", use_container_width=True)
                if yp.exists():
                    st.image(bgr2rgb(cv2.imread(str(yp))), caption="YOLO物体検出", use_container_width=True)

# ============ UI: テストモード ============ 
def ui_mode_test():
    st.header("📦 テストモード（バッチで画像ペアを比較）")
    colA, colB = st.columns([2,1])
    with colA:
        st.markdown("**画像ディレクトリ**")
        default_dir = "test_images"
        test_dir = st.text_input("検索ディレクトリ", value=default_dir, help="PNG/JPG を探索します")
        paths = []
        if test_dir and Path(test_dir).exists():
            paths = sorted([p for p in glob.glob(str(Path(test_dir)/"*.png"))] + [p for p in glob.glob(str(Path(test_dir)/"*.jpg"))] + [p for p in glob.glob(str(Path(test_dir)/"*.jpeg"))])
        if not paths:
            st.warning("ディレクトリ内に画像が見つかりません。")
        else:
            st.caption(f"{len(paths)} 枚見つかりました")

    with colB:
        st.markdown("**クイック追加（よく使う組み合わせ）**")
        candidates = []
        def has(name): return any(Path(p).name == name for p in paths)
        if has("closedoor.png") and has("opendoor.png"):
            candidates.append(("closedoor.png","opendoor.png"))
        if has("closedoor.png") and has("closedoor_different_colors.png"):
            candidates.append(("closedoor.png","closedoor_different_colors.png"))
        if has("easy.png") and has("easy_wrong.png"):
            candidates.append(("easy.png","easy_wrong.png"))
        if has("IMG_4726.PNG") and has("IMG_4728.PNG"):
            candidates.append(("IMG_4726.PNG","IMG_4728.PNG"))
        if candidates:
            for a,b in candidates:
                if st.button(f"＋ 追加: {a} vs {b}", use_container_width=True):
                    pa = str(Path(test_dir)/a); pb = str(Path(test_dir)/b)
                    if (pa,pb) not in ss.test_pairs:
                        ss.test_pairs.append((pa,pb))
                        st.toast(f"追加: {a} vs {b}")
                        st.rerun()
        else:
            st.caption("既知の組み合わせ候補は見つかりませんでした。")

    st.markdown("---")
    st.markdown("**手動でペアを追加**")
    c1, c2, c3 = st.columns([5,5,2])
    with c1:
        selA = st.selectbox("A（基準）", options=["(選択)"]+paths, index=0, key="selA")
    with c2:
        selB = st.selectbox("B（比較）", options=["(選択)"]+paths, index=0, key="selB")
    with c3:
        if st.button("追加", use_container_width=True, disabled=not(selA!='(選択)' and selB!='(選択)')):
            pair = (selA, selB)
            if pair not in ss.test_pairs:
                ss.test_pairs.append(pair)
                st.toast("ペアを追加しました")
                st.rerun()

    st.markdown("---")
    st.subheader("⚙️ 設定（テスト用）")
    cfg = {
        "interval": 0,
        "target_short": st.slider("共通ダウンサンプル短辺（px）", 360, 1440, 720, 60, key="t_target"),
        "align_mode": st.selectbox("整列モード", ["AUTO","ECC","H","OFF"], index=0, key="t_align"),
        "shift_px_thresh": st.slider("ズレ判定しきい値(px)", 1.0, 20.0, 6.0, 0.5, key="t_shift"),
        "min_wh": st.slider("最小ボックス幅/高さ(px)", 5, 100, 15, 1, key="t_minwh"),
        # YOLO（テスト時も可視化）
        "yolo_conf": st.slider("YOLO信頼度しきい値", 0.1, 0.9, float(ss.get("yolo_conf",0.5)), 0.05, key="t_yconf")
    }
    # ▼ テスト設定のヘルプ
    _help_expander(*_HELP_MD["target_short"])
    _help_expander(*_HELP_MD["shift_px_thresh"])
    _help_expander(*_HELP_MD["min_wh"])
    _help_expander(*_HELP_MD["yolo_conf"])

    st.markdown("---")
    run_col1, run_col2 = st.columns([1,4])
    out_root = Path("artifacts_gui") / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if run_col1.button("▶︎ 選択ペアを実行", type="primary", use_container_width=True, disabled=not ss.test_pairs):
        results = []
        for (pa, pb) in ss.test_pairs:
            case_dir = out_root / (Path(pa).stem + "_vs_" + Path(pb).stem)
            r = run_test_case(pa, pb, cfg, case_dir)
            results.append(r)
        st.session_state["test_results"] = results
        st.session_state["test_out_root"] = str(out_root)
        st.success("テスト実行が完了しました。下に結果が表示されます。")

    res = st.session_state.get("test_results")
    if res:
        st.subheader("📊 テスト結果")
        for r in res:
            st.markdown("---")
            if not r.get("ok"):
                st.error(r.get("error", "不明なエラー"))
                continue
            st.markdown(f"**A → B**  \n`{r['A']}`  \n`{r['B']}`")
            cols = st.columns([2,2,2,2,2])
            case_dir = Path(r["dir"])
            with cols[0]:
                st.image(bgr2rgb(cv2.imread(str(case_dir/"A_baseline.png"))), caption="A（基準）", use_container_width=True)
            with cols[1]:
                st.image(bgr2rgb(cv2.imread(str(case_dir/"B_compare.png"))), caption="B（比較）", use_container_width=True)
            with cols[2]:
                st.image(bgr2rgb(cv2.imread(str(case_dir/"diff_boxes.png"))), caption="差分（赤枠・小）", use_container_width=True)
            with cols[3]:
                st.image(bgr2rgb(cv2.imread(str(case_dir/"diff_boxesx2.png"))), caption="差分（大枠クラスタ）", use_container_width=True)
            with cols[4]:
                st.image(cv2.imread(str(case_dir/"diff_mask.png"), cv2.IMREAD_GRAYSCALE), clamp=True, caption="差分マスク", use_container_width=True)
            ssim_txt = "-" if r["ssim"] in (None, -1) else f"{r['ssim']:.4f}"
            st.caption(f"整列: {r['aligned']} / SSIM: {ssim_txt} / 差分率: {r['diff_ratio']*100:.2f}% / ボックス数: {r['boxes']}")
        with st.expander("📦 成果物"):
            st.code(st.session_state.get("test_out_root",""))

# ============ ルーティング（フェーズ遷移） ============
def main():
    st.caption(f"Auto-refresh: {_AUTOREFRESH_IMPL}  •  セッション: `{ss.session_dir or '未作成'}`")
    if ss.mode == "テスト":
        ui_mode_test(); return
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
        _init_state(); ui_step_baseline()

if __name__ == "__main__":
    main()