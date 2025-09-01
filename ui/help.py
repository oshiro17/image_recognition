# ui/help.py
# -*- coding: utf-8 -*-
import streamlit as st

__all__ = ["help_expander", "help_expander_if", "HELP_MD"]

def help_expander(title: str, body_md: str, default_open: bool = False):
    """基本のヘルプ表示"""
    with st.expander("❓ " + title, expanded=default_open):
        st.markdown(body_md)

def help_expander_if(arg1, body_md: str | None = None, default_open: bool = False):
    """
    いずれの呼び方にも対応:
      - help_expander_if("min_wh")                          ← キーだけ
      - help_expander_if(*HELP_MD["min_wh"])                ← タプル展開
      - help_expander_if("タイトル", "本文", default_open)    ← 直接指定

    ※ キー文字列が HELP_MD に無い場合は何も表示せず黙って return します（落ちません）
    """
    # ① キーだけ渡された場合
    if body_md is None and isinstance(arg1, str):
        # キーが登録されていれば表示、無ければ何もしない（安全）
        pair = HELP_MD.get(arg1)
        if pair:
            title, body = pair
            return help_expander(title, body, default_open)
        return  # 未登録キーは無視して終了

    # ② タイトル＋本文（互換）
    if body_md is not None:
        return help_expander(arg1, body_md, default_open)

    # ③ 想定外の形
    return  # 何もしない（落とさない）

# 共通ヘルプ本文
HELP_MD = {
    "baseline_step": (
        "ステップ①：基準を撮影/読み込み",
        """
- 監視の基準となる1枚をカメラ撮影 or 画像アップロードで確定します。
- 基準確定後、この基準に対して「現在フレーム」を定期取得し差分を見ます。
- 基準確定時に YOLO 物体検出を実行し、検出リストを保存します。
"""
    ),
    "camera_backend": (
        "カメラのバックエンド選択",
        """
- **AUTO**: まず自動で開きに行きます。ダメなら他の方式を試してください。
- **AVFOUNDATION / QT**: macOS のカメラ向け。M1/M2/M3環境は **AVFOUNDATION** が安定しやすいです。
- **V4L2**: Linux の一般的なWebカメラインターフェース。
- **DSHOW**: Windows の古い DirectShow 経由。
- 起動できない/緑画面/真っ黒などは **デバイス番号** と **バックエンド** を切り替えて再試行してください。
"""
    ),
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