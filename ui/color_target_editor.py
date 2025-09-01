# ui/color_target_editor.py
import cv2, numpy as np
import streamlit as st
from services.camera import bgr2rgb
from ui.help import help_expander_if

# クリックピック対応（無ければフォールバック）
try:
    from st_click_detector import click_detector as _click_detector
    _HAS_CLICK = True
except Exception:
    _HAS_CLICK = False

def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

def draw_target_preview(img_rgb: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    vis = img_rgb.copy()
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

def _target_mask_for_img(img: np.ndarray, target: dict, roi=None) -> np.ndarray:
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
    roi = target.get("roi")
    m = _target_mask_for_img(img, target, roi)
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        denom = max(1, w * h)
        return float((m[y:y+h, x:x+w] > 0).sum()) / float(denom)
    else:
        return float((m > 0).sum()) / float(m.size)

def render_color_target_editor(ss):
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
    with cols[1]:
        name = st.text_input("名前", value=f"target_{len(ss.targets)+1}", key="t_name")
        direction = st.selectbox("アラート条件", ["decrease", "increase"], help="decrease: 減ったら警告 / increase: 増えたら警告", key="t_dir")
        threshold_pct = st.slider("変化しきい値(%)", 0, 100, 10, key="t_thr")
        help_expander_if("color_threshold")
        k_sigma = st.slider("色ゆるさ kσ", 1.0, 4.0, 2.5, 0.1, key="t_ks")
        help_expander_if("color_k_sigma")
        use_l = st.checkbox("明度Lも使う", value=False, help="色味メインで十分ならOFF推奨", key="t_useL")
        roi_enable = st.checkbox("ROIを指定する", value=False, key="t_roi_en")
        roi = None
        if roi_enable:
            rx = st.slider("ROI x", 0, w-1, 0, key="t_rx")
            ry = st.slider("ROI y", 0, h-1, 0, key="t_ry")
            rw = st.slider("ROI w", 1, w-rx, w, key="t_rw")
            rh = st.slider("ROI h", 1, h-ry, h, key="t_rh")
            roi = (rx, ry, rw, rh)
        if st.button("＋ この色をターゲットに追加", type="primary", use_container_width=True, key="t_add"):
            stats = sample_lab_stats(base, x, y, r, use_l=use_l)
            tmp_target = {"name": name, "mean": stats["mean"], "std": stats["std"], "use_l": use_l,
                          "k_sigma": float(k_sigma), "direction": direction, "threshold_pct": float(threshold_pct), "roi": roi}
            base_ratio = ratio_for_target(base, tmp_target)
            tmp_target["base_ratio"] = float(base_ratio)
            ss.targets.append(tmp_target)
            st.success(f"追加しました: {name} (基準割合 {base_ratio*100:.2f}%)")
        show_mask_preview = st.checkbox("この設定でマスクをプレビュー", value=False, key="t_preview")
        if show_mask_preview:
            tmp = {"name": name, "k_sigma": float(k_sigma), "use_l": use_l,
                   "mean": sample_lab_stats(base, x, y, r, use_l)["mean"],
                   "std": sample_lab_stats(base, x, y, r, use_l)["std"],
                   "roi": roi}
            pm = _target_mask_for_img(base, tmp, roi)
            st.image(pm, clamp=True, caption="ターゲットマスク（基準画像）", use_container_width=True)
    if ss.targets:
        st.markdown("#### 追加済みターゲット")
        del_idxs = []
        for i, t in enumerate(ss.targets):
            cols2 = st.columns([3,2,2,2,1])
            cols2[0].write(f"**{t['name']}**")
            cols2[1].write(f"条件: {t['direction']} / しきい値 {t['threshold_pct']}%")
            cols2[2].write(f"kσ={t.get('k_sigma',2.5)} / L使用={t.get('use_l',False)}")
            cols2[3].write(f"基準割合: {t.get('base_ratio',0.0)*100:.2f}%")
            if cols2[4].button("🗑️", key=f"cfg_del_t_{i}"):
                del_idxs.append(i)
        for i in reversed(del_idxs):
            ss.targets.pop(i)