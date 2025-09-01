# ui/baseline.py
import sys
import cv2, json, numpy as np
import streamlit as st
from pathlib import Path
from services.camera import bgr2rgb, capture_frame
from services.io import new_session_dir
from ui.color_target_editor import render_color_target_editor
from ui.help import help_expander_if, HELP_MD
from core.yolo import detect_objects, draw_detections

def ui_step_baseline(ss):
    st.header("① 基準を撮影 / 読み込み")
    help_expander_if("baseline_step")
    colL, colR = st.columns(2)

    with colL:
        st.markdown("#### 📸 カメラで撮影")
        help_expander_if("camera_backend")
        idx = st.number_input("カメラデバイス番号", 0, 10, ss.get("cam_index",0), 1, key="k_cam_index")

        # ✅ 内部APIを使わず、標準のsys.platformで判定
        default_backend_index = 1 if sys.platform == "darwin" else 0
        be  = st.selectbox(
            "バックエンド",
            ["AUTO","AVFOUNDATION","QT","V4L2","DSHOW"],
            index=default_backend_index,
            key="k_backend"
        )

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
                # YOLO: 基準検出
                yolo_conf = float(ss.get("yolo_conf", 0.5))
                det_base = detect_objects(frame, conf=yolo_conf)
                ss.baseline_yolo = det_base
                with open(Path(session)/"baseline_yolo.json","w",encoding="utf-8") as f:
                    json.dump(det_base, f, ensure_ascii=False, indent=2)
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
        render_color_target_editor(ss)

        with st.expander("🧠 基準のYOLO検出（参考）", expanded=False):
            st.caption(f"しきい値 conf ≥ {ss.get('yolo_conf',0.5)}")
            if Path(ss.session_dir, "baseline_yolo_vis.png").exists():
                st.image(bgr2rgb(cv2.imread(str(Path(ss.session_dir, "baseline_yolo_vis.png")))), use_container_width=True)
            st.json(ss.get("baseline_yolo", []))