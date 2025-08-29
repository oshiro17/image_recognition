# services/state.py
import sys
import streamlit as st

def init_state():
    ss = st.session_state
    ss.setdefault("mode", "監視")
    ss.setdefault("test_pairs", [])
    ss.setdefault("phase", "INIT")          # INIT -> BASELINE_SET -> CONFIG_SET -> RUNNING/PAUSED
    ss.setdefault("session_dir", "")
    ss.setdefault("baseline_path", "")
    ss.setdefault("config", {})
    ss.setdefault("next_shot_ts", 0.0)
    ss.setdefault("cam_diag", {})
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
    return ss