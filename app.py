# app.py
# -*- coding: utf-8 -*-

import streamlit as st
from services.state import init_state
from ui.baseline import ui_step_baseline
from ui.config import ui_step_config
from ui.run import ui_step_run
from ui.testmode import ui_mode_test

st.set_page_config(page_title="差分監視 📸", layout="wide")

# 共通スタイル
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

ss = init_state()

# サイドバー：モード
mode = st.sidebar.radio("モード", ["監視", "テスト"], index=(0 if ss.get("mode","監視")=="監視" else 1))
ss.mode = mode

st.caption(f"セッション: `{ss.session_dir or '未作成'}`")

if ss.mode == "テスト":
    ui_mode_test(ss)
else:
    phase = ss.get("phase","INIT")
    if phase == "INIT":
        ui_step_baseline(ss)
    elif phase == "BASELINE_SET":
        st.success("基準画像 OK！つぎは設定です。")
        ui_step_config(ss)
    elif phase == "CONFIG_SET":
        st.info("設定が保存されました。スタートできます！")
        ui_step_run(ss)
    elif phase in ("RUNNING", "PAUSED"):
        ui_step_run(ss)
    else:
        st.warning("未知の状態です。初期化します。")
        for k in list(ss.keys()): del ss[k]
        init_state()
        ui_step_baseline(ss)