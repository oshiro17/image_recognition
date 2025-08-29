# ui/config.py
import streamlit as st
from services.io import save_config
from ui.color_target_editor import render_color_target_editor
from ui.help import help_expander, HELP_MD

def ui_step_config(ss):
    st.header("② 設定")
    with st.form("cfg_form", clear_on_submit=False):
        st.markdown("**撮影**")
        interval = st.slider("撮影間隔（秒）⏱️", 10, 600, 60, 5)
        target_short = st.slider("共通ダウンサンプル短辺（px）", 360, 1440, 720, 60)
        help_expander(*HELP_MD["target_short"])
        st.markdown("---")
        st.markdown("**整列と検出**")
        align_mode = st.selectbox("整列モード", ["AUTO","ECC","H","OFF"], help="AUTO推奨")
        shift_thresh = st.slider("ズレ判定しきい値(px)", 1.0, 20.0, 6.0, 0.5)
        help_expander(*HELP_MD["shift_px_thresh"])
        min_wh = st.slider("最小ボックス幅/高さ(px)", 5, 100, 15, 1)
        help_expander(*HELP_MD["min_wh"])
        st.markdown("---")
        st.markdown("**YOLO（物体監視）**")
        yolo_conf = st.slider("YOLO信頼度しきい値", 0.1, 0.9, float(ss.get("yolo_conf",0.5)), 0.05)
        help_expander(*HELP_MD["yolo_conf"])
        
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
        render_color_target_editor(ss)