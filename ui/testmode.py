# ui/testmode.py
import glob, cv2
import streamlit as st
from pathlib import Path
from services.camera import bgr2rgb
from logic.compare import compare_to_baseline
# 先頭のimport群に追加
from ui.help import help_expander, HELP_MD

def _imread_color(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def _run_test_case(pathA: str, pathB: str, cfg, out_dir: Path, baseline_yolo):
    a = _imread_color(pathA); b = _imread_color(pathB)
    if a is None or b is None:
        return {"ok": False, "error": f"読込失敗 A={pathA}, B={pathB}"}
    res = compare_to_baseline(a, b, cfg, baseline_yolo)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir/"A_baseline.png"), a)
    cv2.imwrite(str(out_dir/"B_compare.png"), b)
    cv2.imwrite(str(out_dir/"diff_boxes.png"), res["vis_boxes"])
    cv2.imwrite(str(out_dir/"diff_mask.png"), res["mask"])
    cv2.imwrite(str(out_dir/"diff_boxesx2.png"), res["clusters_vis"])
    if isinstance(res.get("yolo_vis"), (list, tuple)) is False and res.get("yolo_vis") is not None:
        cv2.imwrite(str(out_dir/"yolo_vis.png"), res["yolo_vis"])
    return {
        "ok": True, "A": pathA, "B": pathB,
        "aligned": res["aligned"], "ssim": res["ssim"],
        "diff_ratio": res["diff_ratio"], "boxes": len(res["boxes"]),
        "dir": str(out_dir)
    }

def ui_mode_test(ss):
    st.header("📦 テストモード（バッチで画像ペアを比較）")
    colA, colB = st.columns([2,1])
    with colA:
        st.markdown("**画像ディレクトリ**")
        default_dir = "test_images"
        test_dir = st.text_input("検索ディレクトリ", value=default_dir, help="PNG/JPG を探索します")
        paths = []
        if test_dir and Path(test_dir).exists():
            paths = sorted([p for p in glob.glob(str(Path(test_dir)/"*.png"))] +
                           [p for p in glob.glob(str(Path(test_dir)/"*.jpg"))] +
                           [p for p in glob.glob(str(Path(test_dir)/"*.jpeg"))])
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
        "yolo_conf": st.slider("YOLO信頼度しきい値", 0.1, 0.9, float(ss.get("yolo_conf",0.5)), 0.05, key="t_yconf")
    }
    help_expander(*HELP_MD["target_short"])
    help_expander(*HELP_MD["shift_px_thresh"])
    help_expander(*HELP_MD["min_wh"])
    help_expander(*HELP_MD["yolo_conf"])

    st.markdown("---")
    run_col1, run_col2 = st.columns([1,4])
    out_root = Path("artifacts_gui") / __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if run_col1.button("▶︎ 選択ペアを実行", type="primary", use_container_width=True, disabled=not ss.test_pairs):
        results = []
        for (pa, pb) in ss.test_pairs:
            case_dir = out_root / (Path(pa).stem + "_vs_" + Path(pb).stem)
            r = _run_test_case(pa, pb, cfg, case_dir, ss.get("baseline_yolo", []))
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