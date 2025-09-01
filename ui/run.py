# ui/run.py
import time, glob, cv2
import numpy as np
import streamlit as st
from pathlib import Path
from services.camera import bgr2rgb, capture_frame
from services.io import append_index, make_zip
from services.alerts import alert_banner
from logic.compare import compare_to_baseline
from ui.help import help_expander_if

# 自動リロード（パッケージが無ければJSでフォールバック）
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_IMPL = "pkg"
except Exception:
    from streamlit.components.v1 import html as _html
    def st_autorefresh(interval: int = 5_000, key: str = "tick", limit: int | None = None):
        _html(f"<script>setTimeout(()=>window.location.reload(), {interval});</script>", height=0)
        return None
    _AUTOREFRESH_IMPL = "js"

def imread_color(path: str):
    if not path: return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img if isinstance(img, np.ndarray) and img.size > 0 else None

def ui_step_run(ss):
    st.header("③ 監視スタート")
    help_expander_if("run_step")
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
                res = compare_to_baseline(baseline, frame, ss.config, ss.get("baseline_yolo", []))

                tsname = __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
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
                ss.last_yolo_changes = res.get("yolo_changes", {})

                # アラートまとめ
                msgs = []
                yc = res.get("yolo_changes", {})
                if ss.config.get("yolo_watch_person", True) and yc.get("person_appeared"):
                    msgs.append("人が現れました！")
                if ss.config.get("yolo_alert_new", True) and yc.get("new_labels"):
                    msgs.append("新しい物体: " + ", ".join(yc["new_labels"]))
                if ss.config.get("yolo_alert_missing", True) and yc.get("missing_labels"):
                    msgs.append("消えた物体: " + ", ".join(yc["missing_labels"]))
                # YOLO: 面積変化/移動量のアラート
                for a in (yc.get("area_alerts") or []):
                    try:
                        lbl = a.get("label", "obj")
                        dp = float(a.get("delta_pct", 0))
                        msgs.append(f"{lbl} のサイズ変化 {dp:+.1f}% (基準 {a.get('base_area',0):.0f} → 現在 {a.get('curr_area',0):.0f})")
                    except Exception:
                        pass
                for m in (yc.get("moved_alerts") or []):
                    try:
                        lbl = m.get("label", "obj")
                        shift = float(m.get("shift_px", 0))
                        msgs.append(f"{lbl} が移動しました (≈ {shift:.1f}px)")
                    except Exception:
                        pass
                for t in (res.get("target_results", []) or []):
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
                    alert_banner(" / ".join(msgs))

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

            # YOLO変化の要約テーブル
            with st.expander("🧠 YOLO 変化ログ", expanded=False):
                yc_last = ss.get("last_yolo_changes") or {}
                try:
                    import pandas as pd
                    rows = []
                    for a in (yc_last.get("area_alerts") or []):
                        rows.append({
                            "type": "area",
                            "label": a.get("label"),
                            "delta(%)": round(float(a.get("delta_pct", 0)), 1),
                            "base_area": int(a.get("base_area", 0)),
                            "curr_area": int(a.get("curr_area", 0)),
                        })
                    for m in (yc_last.get("moved_alerts") or []):
                        rows.append({
                            "type": "move",
                            "label": m.get("label"),
                            "shift_px": round(float(m.get("shift_px", 0)), 1),
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        st.caption("しきい値を超えるYOLO変化はありませんでした。")
                except Exception:
                    for a in (yc_last.get("area_alerts") or []):
                        st.write(f"• [area] {a.get('label','obj')} Δ {a.get('delta_pct',0):.1f}% (基準 {a.get('base_area',0)} → 現在 {a.get('curr_area',0)})")
                    for m in (yc_last.get("moved_alerts") or []):
                        st.write(f"• [move] {m.get('label','obj')} shift ≈ {m.get('shift_px',0):.1f}px")

            with st.expander("🎯 色ターゲットの結果", expanded=False):
                tr = ss.get("last_target_results", [])
                if tr:
                    try:
                        import pandas as pd
                        rows = []
                        for t in tr:
                            if "error" in t:
                                rows.append({"name": t.get("name","target"), "error": t["error"]})
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

        lm = ss.get("last_metrics") or {}
        info.info(
            f"整列: {lm.get('aligned','-')} / "
            f"SSIM: {('-' if lm.get('ssim') in (None,-1) else f'{lm.get('ssim'):.4f}')} / "
            f"差分率: {lm.get('diff_ratio','-') if isinstance(lm.get('diff_ratio'),str) else f'{(lm.get('diff_ratio') or 0)*100:.2f}%'} / "
            f"ボックス数: {lm.get('boxes','-')}"
        )