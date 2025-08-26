import time
import cv2
import numpy as np
import streamlit as st
import sys

# 自動リロード機能：streamlit-autorefresh が無ければ簡易JSにフォールバック
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
    _AUTOREFRESH_IMPL = "pkg"
except Exception:  # ModuleNotFoundError 等
    from streamlit.components.v1 import html as _html
    def st_autorefresh(interval: int = 5_000, key: str = "tick", limit: int | None = None):
        # 非公式の簡易版：JSでページをリロードする。limitは未対応だがPoC用途では十分。
        _html(f"<script>setTimeout(() => window.location.reload(), {interval});</script>", height=0)
        return None
    _AUTOREFRESH_IMPL = "js"

from make_ref import (
    ensure_same_size, camera_misaligned, difference_boxes, fused_diff_mask, boxes_from_mask,
    make_boxes_overlay_transparent, cluster_dense_boxes, draw_clusters_only,
    align_ecc, align_homography
)

st.set_page_config(page_title="定期撮影差分（1分ごと）", layout="wide")
st.title("📷 定期撮影差分 PoC（1分ごとに前回と比較）")
st.caption(f"Auto-refresh: {_AUTOREFRESH_IMPL}")

# --- 設定 ---
interval_sec = st.sidebar.slider("撮影間隔（秒）", 10, 300, 60, 10)  # 1分=60秒
shift_px_thresh = st.sidebar.slider("ズレ判定しきい値(px)", 1.0, 20.0, 6.0, 0.5)
min_wh = st.sidebar.slider("最小ボックス幅/高さ(px)", 5, 100, 15, 1)

cam_index = st.sidebar.number_input("カメラデバイス番号", min_value=0, max_value=10, value=0, step=1)
backend_name = st.sidebar.selectbox(
    "バックエンド(キャプチャAPI)",
    options=["AUTO", "AVFOUNDATION", "QT", "V4L2", "DSHOW"],
    index=1 if sys.platform == "darwin" else 0,
    help="macはAVFOUNDATIONが安定。LinuxはV4L2、WindowsはDSHOW/MediaFoundation"
)

run = st.sidebar.toggle("開始 / 停止", value=False)
st.sidebar.caption("開始中はタブを開いたままにしてください（PoC用途）")

# --- セッション状態 ---
ss = st.session_state
ss.setdefault("prev_img", None)
ss.setdefault("prev_ts", 0.0)
ss.setdefault("last_result", None)  # 直近の出力キャッシュ

# 定期的に再実行（UIを固まらせないPoC手法）
st_autorefresh_token = st_autorefresh(interval=5_000, key="tick")  # 5秒ごとに再実行

def _backend_flag(name: str) -> int | None:
    name = (name or "AUTO").upper()
    if name == "AUTO":
        return None
    return {
        "AVFOUNDATION": getattr(cv2, "CAP_AVFOUNDATION", None),
        "QT": getattr(cv2, "CAP_QT", None),
        "V4L2": getattr(cv2, "CAP_V4L2", None),
        "DSHOW": getattr(cv2, "CAP_DSHOW", None),
    }.get(name)


def capture_frame(index: int, backend: str, warmup: int = 5, width: int | None = None, height: int | None = None) -> tuple[np.ndarray | None, dict]:
    """カメラから1フレーム取得。失敗時は別バックエンドでリトライ。
    戻り値: (frame, diag)
    diagには opened, backend_used, try_order, shape などを入れる。
    """
    tries = []
    # 試行順序を決める
    order = [backend]
    if backend != "AUTO":
        order.append("AUTO")
    # mac の場合はAVFOUNDATION→QTも試す
    if sys.platform == "darwin":
        for alt in ("AVFOUNDATION", "QT"):
            if alt not in order:
                order.append(alt)
    # それ以外のOS
    else:
        for alt in ("V4L2", "DSHOW"):
            if alt not in order:
                order.append(alt)

    for b in order:
        flag = _backend_flag(b)
        cap = cv2.VideoCapture(index, flag) if flag is not None else cv2.VideoCapture(index)
        opened = cap.isOpened()
        diag = {"opened": opened, "backend_used": b, "try_order": order}
        if not opened:
            cap.release()
            tries.append((b, False))
            continue
        # 解像度指定（任意）
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # ウォームアップで数フレーム捨てる
        for _ in range(max(0, warmup)):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.size > 0:
            diag["shape"] = tuple(frame.shape)
            return frame, diag
        tries.append((b, True))
    # 全て失敗
    return None, {"opened": False, "backend_used": None, "try_order": order, "shape": None}

def bgr2rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- メイン処理 ---
placeholder_info = st.empty()
col1, col2 = st.columns(2)
ph_prev = col1.empty()
ph_curr = col2.empty()

if run:
    now = time.time()
    need_shoot = (now - ss["prev_ts"] >= interval_sec) or (ss["prev_img"] is None)

    if need_shoot:
        curr, cam_diag = capture_frame(cam_index, backend_name)
        if curr is None:
            st.error("カメラが見つからない/アクセスできません。以下を確認してください：\n- macOSのシステム設定 > プライバシーとセキュリティ > カメラ で、使用中のターミナル/IDEに許可を付与\n- デバイス番号（左の\"カメラデバイス番号\"）を変更して再試行\n- バックエンドを切り替え（AVFOUNDATION/QT/V4L2/DSHOW）")
            st.code(str(cam_diag))
        else:
            # 表示用
            ph_curr.image(bgr2rgb(curr), caption="今の写真", use_column_width=True)
            st.caption(f"Camera: index={cam_index}, backend={cam_diag.get('backend_used')} shape={cam_diag.get('shape')}")

            if ss["prev_img"] is not None:
                imgA, imgB = ensure_same_size(ss["prev_img"], curr)

                # 1) ズレ判定
                misaligned, diag = camera_misaligned(imgA, imgB, shift_px_thresh=shift_px_thresh, homography_checks=True)
                aligned_method = None
                if misaligned:
                    b_ecc = align_ecc(imgA, imgB)
                    if b_ecc is not None:
                        imgB = b_ecc; aligned_method = "ECC"
                    else:
                        b_h = align_homography(imgA, imgB)
                        if b_h is not None:
                            imgB = b_h; aligned_method = "H"
                        else:
                            placeholder_info.warning("自動整列に失敗したため、今回の差分はスキップしました。")
                            # 前回画像は更新して終了
                            ss["prev_img"] = curr; ss["prev_ts"] = now
                            st.stop()

                # 2) 差分抽出（厳密一致ならabsdiff、そうでなければ照明に強いマスク）
                strict_same = (aligned_method is None) and (diag.get("shift_norm", 0.0) <= 2.0)
                if strict_same:
                    vis_boxes, th_mask, boxes = difference_boxes(imgA, imgB, min_wh=min_wh, bin_thresh=None)
                else:
                    th_mask = fused_diff_mask(imgA, imgB)
                    vis_boxes, boxes = boxes_from_mask(imgA, th_mask, min_wh=min_wh)

                area_ratio = float((th_mask > 0).sum()) / float(th_mask.size)
                overlay = make_boxes_overlay_transparent(imgA, imgB, boxes, alpha=0.5, draw_border=True) if boxes else imgA.copy()
                clusters = cluster_dense_boxes(boxes, imgA.shape, dilate_iter=3, min_count=6, min_wh=30)
                clusters_vis = draw_clusters_only(imgA, clusters, color=(0,0,255), thickness=6, show_count=True)

                # 3) 画面表示
                ph_prev.image(bgr2rgb(imgA), caption="1分前の写真", use_column_width=True)

                st.subheader("差分結果")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(bgr2rgb(vis_boxes), caption="赤枠（差分矩形）", use_column_width=True)
                with c2:
                    st.image(th_mask, clamp=True, caption="差分マスク（二値）", use_column_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    st.image(bgr2rgb(overlay), caption="半透明合成", use_column_width=True)
                with c4:
                    st.image(bgr2rgb(clusters_vis), caption="密集クラスタ（大枠）", use_column_width=True)

                aligned_txt = aligned_method if aligned_method else "なし"
                placeholder_info.info(
                    f"位相相関: dx={diag.get('phase_shift',(0,0))[0]:.2f}, "
                    f"dy={diag.get('phase_shift',(0,0))[1]:.2f}, "
                    f"norm={diag.get('shift_norm',0.0):.2f}px / 整列: {aligned_txt} / "
                    f"差分面積比: {area_ratio*100:.2f}% / ボックス数: {len(boxes)}"
                )

                # キャッシュ
                ss["last_result"] = {
                    "diag": diag, "aligned": aligned_method, "area_ratio": area_ratio, "boxes": len(boxes)
                }

            # 更新：今回を「前回」に
            ss["prev_img"] = curr
            ss["prev_ts"] = now

else:
    st.info("開始をオンにすると、1分ごとに自動撮影して直前のフレームと比較します。")

# 直近サマリ
if ss.get("last_result"):
    with st.expander("直近の処理サマリ"):
        st.json(ss["last_result"])