# services/camera.py
from typing import Optional, Tuple, Dict, Any
import cv2, numpy as np, sys

def _backend_flag(name: str) -> Optional[int]:
    m = {
        "AVFOUNDATION": getattr(cv2, "CAP_AVFOUNDATION", None),
        "QT": getattr(cv2, "CAP_QT", None),
        "V4L2": getattr(cv2, "CAP_V4L2", None),
        "DSHOW": getattr(cv2, "CAP_DSHOW", None),
        "AUTO": None,
    }
    return m.get((name or "AUTO").upper())

def capture_frame(index: int, backend: str, warmup: int = 5, size: Tuple[int,int] | None=None) -> Tuple[Optional[np.ndarray], Dict[str,Any]]:
    order = [backend] + [b for b in (("AVFOUNDATION","QT") if sys.platform=="darwin" else ("V4L2","DSHOW")) if b!=backend] + (["AUTO"] if backend!="AUTO" else [])
    for b in order:
        flag = _backend_flag(b)
        cap = cv2.VideoCapture(index, flag) if flag is not None else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release(); continue
        if size:
            w,h = size; cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(max(0,warmup)): cap.read()
        ok, frame = cap.read(); cap.release()
        if ok and isinstance(frame, np.ndarray) and frame.size>0:
            return frame, {"backend_used": b, "shape": tuple(frame.shape)}
    return None, {"backend_used": None, "shape": None}

def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)