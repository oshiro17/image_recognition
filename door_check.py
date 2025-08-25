# door_check.py
import argparse, os, cv2, numpy as np
from skimage.metrics import structural_similarity as ssim

def imread(p):  # 日本語パス対応
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
def imwrite(p, img):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imencode(os.path.splitext(p)[1], img)[1].tofile(p)

def ensure_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
def resize_to(img, max_wh=(1280, 720)):
    h, w = img.shape[:2]; sx = max_wh[0]/w; sy = max_wh[1]/h; s = min(sx, sy, 1.0)
    return cv2.resize(img, (int(w*s), int(h*s))) if s<1.0 else img

def lab_ab_hist(bgr, bins=48):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); a, b = lab[:,:,1], lab[:,:,2]
    hist = cv2.calcHist([a,b],[0,1],None,[bins,bins],[0,256,0,256])
    return cv2.normalize(hist, hist).astype("float32")

def scene_match_score(img, ref):
    # 「色の割合が大まか一致」を測る（Lab a,b ヒスト相関）
    return float(cv2.compareHist(lab_ab_hist(img), lab_ab_hist(ref), cv2.HISTCMP_CORREL))

def stabilize_to_ref(mov, ref):
    orb = cv2.ORB_create(1200); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    gR, gM = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)
    kR,dR = orb.detectAndCompute(gR,None); kM,dM = orb.detectAndCompute(gM,None)
    if dR is None or dM is None or len(kR)<8 or len(kM)<8: return mov
    ms = sorted(bf.match(dR,dM), key=lambda m:m.distance)[:200]
    if len(ms)<8: return mov
    pR = np.float32([kR[m.queryIdx].pt for m in ms]).reshape(-1,1,2)
    pM = np.float32([kM[m.trainIdx].pt for m in ms]).reshape(-1,1,2)
    H,_ = cv2.findHomography(pM, pR, cv2.RANSAC, 4.0)
    return cv2.warpPerspective(mov, H, (ref.shape[1], ref.shape[0])) if H is not None else mov

def auto_roi(img):
    # ドアっぽい縦長領域をざっくり推定（外れたら --roi で上書き）
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g,(5,5),0), 60, 160)
    cnts,_ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = g.shape[:2]; best=None; best_s=-1
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        if hh<h*0.25 or ww<w*0.05: continue
        aspect = hh/max(1,ww)
        if aspect<1.6: continue
        mean_v = g[y:y+hh, x:x+ww].mean()
        score = (aspect-1.6) + (255-mean_v)/120.0 + e[y:y+hh, x:x+ww].mean()/255.0
        if score>best_s: best_s, best = score, (x,y,ww,hh)
    if best is None:  # 右側を仮ROI
        x=int(w*0.65); y=int(h*0.2); ww=int(w*0.3); hh=int(h*0.7)
        best=(x,y,ww,hh)
    return best

def rel_to_abs(img, rel):
    h,w = img.shape[:2]
    x0=int(rel[0]*w); y0=int(rel[1]*h); x1=int(rel[2]*w); y1=int(rel[3]*h)
    return (x0,y0,max(1,x1-x0),max(1,y1-y0))

def diff_maps(ref, cur):
    g1,g2 = ensure_gray(ref), ensure_gray(cur)
    h=min(g1.shape[0],g2.shape[0]); w=min(g1.shape[1],g2.shape[1]); g1,g2=g1[:h,:w],g2[:h,:w]
    diff = cv2.absdiff(g1,g2)
    _,th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    d = np.clip((g1.astype(np.int16)-g2.astype(np.int16))//2 + 128, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
    return diff, th, diff_color

def draw_boxes(base, mask, min_wh=20):
    out = base.copy()
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w>=min_wh or h>=min_wh:
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2)
    return out

def door_open_score(ref_roi, cur_roi):
    g1,g2 = ensure_gray(ref_roi), ensure_gray(cur_roi)
    e1,e2 = cv2.Canny(g1,60,160), cv2.Canny(g2,60,160)
    edge_delta = abs(int(e2.sum())-int(e1.sum()))/(255.0*max(1,e1.size))
    mean_delta = abs(g2.mean()-g1.mean())/255.0
    return 0.7*edge_delta + 0.3*mean_delta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="比較対象（現在）画像")
    ap.add_argument("--ref", required=True, help="参照（閉）画像")
    ap.add_argument("--out", default="out", help="出力先")
    ap.add_argument("--roi", nargs=4, type=float, help="相対ROI x0 y0 x1 y1（任意）")
    ap.add_argument("--scene_corr_thr", type=float, default=0.75, help="同一場面とみなす色相関の閾値")
    ap.add_argument("--open_thr", type=float, default=0.14, help="開状態スコア閾値")
    args = ap.parse_args()

    cur = resize_to(imread(args.img)); ref = resize_to(imread(args.ref))
    if cur is None or ref is None: raise SystemExit("画像が読めません")

    # 1) 同一場面か判定（色の割合）
    scene_corr = scene_match_score(cur, ref)
    print(f"[Scene] Lab-ab相関: {scene_corr:.3f}  (thr={args.scene_corr_thr})  ->",
          "SAME" if scene_corr>=args.scene_corr_thr else "DIFFERENT")
    if scene_corr < args.scene_corr_thr:
        print("※ 色の割合が大きく異なるため、別の場面の可能性が高いです。")
        # ここで停止せず、参考出力は継続

    # 2) 幾何補正
    cur_aligned = stabilize_to_ref(cur, ref)
    imwrite(os.path.join(args.out, "aligned.jpg"), cur_aligned)

    # 3) ROI（相対指定があれば優先、なければ自動）
    if args.roi:
        rx,ry,rw,rh = rel_to_abs(ref, args.roi)
    else:
        rx,ry,rw,rh = auto_roi(ref)
    print(f"[ROI] x={rx} y={ry} w={rw} h={rh}")

    ref_roi = ref[ry:ry+rh, rx:rx+rw]
    cur_roi = cur_aligned[ry:ry+rh, rx:rx+rw]

    # 4) 可視化: 赤枠方式 & 差分画像方式
    diff, th, diff_color = diff_maps(ref_roi, cur_roi)

    # 赤枠方式（ROIを全体に埋め戻して保存）
    boxed = draw_boxes(cur_roi, th, min_wh=max(10, min(rw,rh)//20))
    full = cur_aligned.copy()
    full[ry:ry+rh, rx:rx+rw] = boxed
    cv2.rectangle(full, (rx,ry), (rx+rw,ry+rh), (0,255,0), 2)
    imwrite(os.path.join(args.out, "diff_boxes.jpg"), full)

    # 差分ヒートマップも重ね保存
    heat = cur_aligned.copy()
    heat_roi = cv2.resize(diff_color, (rw, rh))
    heat[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(heat_roi, 0.6, heat[ry:ry+rh, rx:rx+rw], 0.4, 0)
    imwrite(os.path.join(args.out, "diff_map.png"), heat)

    # 5) 開閉スコア
    open_score = door_open_score(ref_roi, cur_roi)
    print(f"[Door] open_score={open_score:.3f} (thr={args.open_thr}) ->",
          "OPEN" if open_score>=args.open_thr else "CLOSED/UNCHANGED")

if __name__ == "__main__":
    main()