# logic/yolo_changes.py
from typing import List, Dict, Any, Optional

def _iou_xywh(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah; bx2, by2 = bx+bw, by+bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = aw*ah + bw*bh - inter
    return float(inter) / float(max(1, union))

def analyze_yolo_changes(baseline_det: List[Dict[str,Any]], current_det: List[Dict[str,Any]], iou_thr: float=0.3):
    def _label_of(d: Dict[str,Any]) -> Optional[str]:
        for k in ("label","name","class_name","cls_name","cls","class"):
            if k in d and d[k] is not None:
                return str(d[k])
        return None

    def _bbox_of(d: Dict[str,Any]) -> Optional[tuple]:
        if "bbox" in d and d["bbox"] is not None:
            x, y, w, h = d["bbox"]; return (float(x), float(y), float(w), float(h))
        if "xywh" in d and d["xywh"] is not None:
            x, y, w, h = d["xywh"]; return (float(x), float(y), float(w), float(h))
        if "xyxy" in d and d["xyxy"] is not None:
            x1, y1, x2, y2 = d["xyxy"]; return (float(x1), float(y1), max(0.0, float(x2)-float(x1)), max(0.0, float(y2)-float(y1)))
        if isinstance(d.get("box"), dict):
            bx = d["box"]
            if all(k in bx for k in ("x","y","w","h")):
                return (float(bx["x"]), float(bx["y"]), float(bx["w"]), float(bx["h"]))
        return None

    base_labels = [lb for lb in (_label_of(d) for d in baseline_det) if lb]
    cur_labels  = [lb for lb in (_label_of(d) for d in current_det) if lb]

    new_labels = sorted(list(set(cur_labels) - set(base_labels)))
    missing_labels = sorted(list(set(base_labels) - set(cur_labels)))

    def _matched(s: Dict[str,Any], t: Dict[str,Any]) -> bool:
        slb, tlb = _label_of(s), _label_of(t)
        if not slb or not tlb or slb != tlb:
            return False
        sbx, tbx = _bbox_of(s), _bbox_of(t)
        if not sbx or not tbx:
            return False
        return _iou_xywh(sbx, tbx) >= iou_thr

    missing_instances = [s for s in baseline_det if not any(_matched(s, t) for t in current_det)]
    appeared_instances = [t for t in current_det if not any(_matched(s, t) for s in baseline_det)]

    person_appeared = ("person" not in base_labels) and ("person" in cur_labels)
    person_missing  = ("person" in base_labels) and ("person" not in cur_labels)

    return {
        "new_labels": new_labels,
        "missing_labels": missing_labels,
        "appeared_instances": appeared_instances,
        "missing_instances": missing_instances,
        "person_appeared": person_appeared,
        "person_missing": person_missing
    }