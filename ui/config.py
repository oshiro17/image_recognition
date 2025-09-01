# ui/config.py
import streamlit as st
from services.io import save_config
from ui.color_target_editor import render_color_target_editor
from ui.help import help_expander, HELP_MD

def ui_step_config(ss):
    st.header("② 設定")
    with st.form("cfg_form", clear_on_submit=False):
        # ================= 撮影 =================
        st.markdown("**撮影**")
        interval = st.slider("撮影間隔（秒）⏱️", 10, 600, 60, 5)
        target_short = st.slider("共通ダウンサンプル短辺（px）", 360, 1440, 720, 60)
        help_expander(*HELP_MD["target_short"])

        st.markdown("---")
        # ================= 整列と検出 =================
        st.markdown("**整列と検出**")
        align_mode = st.selectbox("整列モード", ["AUTO","ECC","H","OFF"], help="AUTO推奨")
        shift_thresh = st.slider("ズレ判定しきい値(px)", 1.0, 20.0, 6.0, 0.5)
        help_expander(*HELP_MD["shift_px_thresh"])
        min_wh = st.slider("最小ボックス幅/高さ(px)", 5, 100, 15, 1)
        help_expander(*HELP_MD["min_wh"])

        st.markdown("---")
        # ================= YOLO（物体監視） =================
        st.markdown("**YOLO（物体監視）**")
        yolo_conf = st.slider("YOLO信頼度しきい値", 0.1, 0.9, float(ss.get("yolo_conf",0.5)), 0.05)
        help_expander(*HELP_MD["yolo_conf"])

        # 物体位置を使った変化検出のパラメータ
        yolo_iou_thr = st.slider(
            "YOLO: マッチ判定IoUしきい値",
            0.1, 0.9, float(ss.get("yolo_iou_thr", 0.30)), 0.05,
            help="基準物体と現在物体を同一とみなす重なり度(IoU)の下限。低いと誤マッチ増・高いと見落とし増"
        )
        if "yolo_iou_thr" in HELP_MD:
            help_expander(*HELP_MD["yolo_iou_thr"])  # あれば詳細を展開

        yolo_obj_diff_thr_pct = st.slider(
            "YOLO: 物体内差分しきい値(%)",
            1.0, 100.0, float(ss.get("yolo_obj_diff_thr_pct", 15.0)), 1.0,
            help="マッチした物体のバウンディングボックス内で、差分がこの割合以上なら『その物体が変化』と判定"
        )
        if "yolo_obj_diff_thr_pct" in HELP_MD:
            help_expander(*HELP_MD["yolo_obj_diff_thr_pct"])  # あれば詳細を展開

        watch_person = st.toggle("人の出現を監視（基準に居なかったのに現れたら通知）", value=bool(ss.get("yolo_watch_person", True)))
        alert_new = st.toggle("基準にないラベルが現れたら通知", value=bool(ss.get("yolo_alert_new", True)))
        alert_missing = st.toggle("基準にあったラベルが消えたら通知", value=bool(ss.get("yolo_alert_missing", True)))

        st.markdown("---")
        # ================= 色ターゲット（共通） =================
        st.markdown("**色ターゲット（共通）**")
        # 相対: 基準比の何％未満で『消滅』扱いにするか
        vanish_rel_pct = st.slider(
            "色ターゲット: 消滅判定の相対下限(%)",
            0.0, 100.0, float(ss.get("target_vanish_rel_pct", 10.0)), 1.0,
            help="基準の画素比×この割合を下回ったら『消滅』とみなす（例: 10%）"
        )
        # 絶対: 画像全体に対する絶対割合％（非常に小さい値向けのセーフガード）
        vanish_abs_pct = st.number_input(
            "色ターゲット: 消滅判定の絶対下限(%)",
            min_value=0.0, max_value=1.0, value=float(ss.get("target_vanish_abs_pct", 0.05)), step=0.01,
            help="どれだけ基準が大きくても、この絶対下限％を下回れば『消滅』とみなす（例: 0.05%）"
        )
        # 小数(0-1)に変換して保存するための内部値
        target_vanish_rel = float(vanish_rel_pct) / 100.0
        target_vanish_abs = float(vanish_abs_pct) / 100.0

        if "target_vanish" in HELP_MD:
            help_expander(*HELP_MD["target_vanish"])  # ドキュメントにある場合のみ

        st.markdown("---")
        submitted = st.form_submit_button("✅ 設定を保存して次へ", use_container_width=True)
        if submitted:
            ss.yolo_conf = float(yolo_conf)
            ss.yolo_watch_person = bool(watch_person)
            ss.yolo_alert_new = bool(alert_new)
            ss.yolo_alert_missing = bool(alert_missing)
            ss.yolo_iou_thr = float(yolo_iou_thr)
            ss.yolo_obj_diff_thr_pct = float(yolo_obj_diff_thr_pct)
            ss.target_vanish_rel_pct = float(vanish_rel_pct)
            ss.target_vanish_abs_pct = float(vanish_abs_pct)

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
                "yolo_iou_thr": float(yolo_iou_thr),
                "yolo_obj_diff_thr_pct": float(yolo_obj_diff_thr_pct),
                # 色ターゲット（共通）
                "target_vanish_rel": float(target_vanish_rel),  # 0-1
                "target_vanish_abs": float(target_vanish_abs),  # 0-1
            }
            save_config(ss.session_dir, ss.config)
            ss.phase = "CONFIG_SET"
            ss.next_shot_ts = 0.0
            st.success("設定を保存しました！")
            st.rerun()

    st.markdown("---")
    with st.expander("🎯 監視する色ターゲットを設定（任意）", expanded=True):
        # 注: 個別ターゲットの『変化しきい値(%)』上限は ui/color_target_editor.py 側で 100% までに引き上げてください
        render_color_target_editor(ss)