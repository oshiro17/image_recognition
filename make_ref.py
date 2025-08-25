# image_recorce/compare_two.py
import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim

IMG1 = "test1b_different_colors.png"  # 参照（基準）
IMG2 = "test1b.png"                    # 比較対象

# 相対ROI(0~1): 右側ドア周辺だけ。必要に応じて微調整
REL_ROI = (0.70, 0.25, 0.96, 0.92)  # (x0,y0,x1,y1) 右の黒枠ドア中心

def rel2abs_roi(img, rel):
    h, w = img.shape[:2]
    x0 = int(rel[0] * w); y0 = int(rel[1] * h)
    x1 = int(rel[2] * w); y1 = int(rel[3] * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w-1, x1), min(h-1, y1)
    return (x0, y0, max(1, x1-x0), max(1, y1-y0))

def reinhard_color_transfer(src_bgr, ref_bgr):
    # Lab空間で平均と分散を一致させる（Reinhard）
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = np.empty_like(src)
    for c in range(3):
        ms, ss = src[:,:,c].mean(), src[:,:,c].std()+1e-6
        mr, sr = ref[:,:,c].mean(), ref[:,:,c].std()+1e-6
        out[:,:,c] = ((src[:,:,c]-ms)*(sr/ss) + mr)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

def stabilize_to_ref(mov_bgr, ref_bgr):
    # ORBで特徴合わせ → mov を ref 座標系へワープ
    orb = cv2.ORB_create(1000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    g_ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    g_mov = cv2.cvtColor(mov_bgr, cv2.COLOR_BGR2GRAY)
    kpR, desR = orb.detectAndCompute(g_ref, None)
    kpM, desM = orb.detectAndCompute(g_mov, None)
    if desR is None or desM is None or len(kpR)<8 or len(kpM)<8:
        return mov_bgr  # そのまま
    matches = sorted(bf.match(desR, desM), key=lambda m: m.distance)[:120]
    if len(matches) < 8: return mov_bgr
    ptsR = np.float32([kpR[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    ptsM = np.float32([kpM[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(ptsM, ptsR, cv2.RANSAC, 4.0)
    if H is None: return mov_bgr
    return cv2.warpPerspective(mov_bgr, H, (ref_bgr.shape[1], ref_bgr.shape[0]))

def edge_map(bgr):
    # VにCLAHE→Sobel勾配→二値化（CannyよりWB影響小さめ）
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v = cv2.createCLAHE(2.0,(8,8)).apply(v)
    g = cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1,0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0,1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = (mag > 60).astype(np.uint8)*255
    return edges

def ab_hist(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:,:,1]; b = lab[:,:,2]
    hist = cv2.calcHist([a,b], [0,1], None, [48,48], [0,256,0,256])
    return cv2.normalize(hist, hist).astype("float32")

def compare(ref_full, mov_full):
    # 幾何補正（mov→ref）
    mov_stab = stabilize_to_ref(mov_full, ref_full)

    # ROI切り出し（相対 → 絶対）
    rx,ry,rw,rh = rel2abs_roi(ref_full, REL_ROI)
    ref = ref_full[ry:ry+rh, rx:rx+rw]
    mov = mov_stab[ry:ry+rh, rx:rx+rw]

    # 色合わせ（mov を ref に近づける）
    mov_ct = reinhard_color_transfer(mov, ref)

    # 形の比較（勾配SSIM）
    e1, e2 = edge_map(ref), edge_map(mov_ct)
    h = min(e1.shape[0], e2.shape[0]); w = min(e1.shape[1], e2.shape[1])
    ssim_grad = ssim(e1[:h,:w], e2[:h,:w])

    # 色の比較（Lab の a,b ヒスト）
    corr_ab = cv2.compareHist(ab_hist(ref), ab_hist(mov_ct), cv2.HISTCMP_CORREL)

    return corr_ab, ssim_grad

if __name__ == "__main__":
    ref = cv2.imread(IMG1); mov = cv2.imread(IMG2)
    if ref is None or mov is None: raise SystemExit("画像を読めません")

    corr_ab, ssim_grad = compare(ref, mov)
    print("=== 改良版・比較結果 ===")
    print(f"abヒスト相関 : {corr_ab:.3f}  (>=0.88 なら色系統OK)")
    print(f"勾配SSIM     : {ssim_grad:.3f} (>=0.75 なら形OK)")

    SAME_COLOR = corr_ab >= 0.88
    SAME_SHAPE = ssim_grad >= 0.75
    print(">>>", "同じ状況と判定（どちらか満たせばOK）" if (SAME_COLOR or SAME_SHAPE) else "違う状況と判定")