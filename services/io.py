# services/io.py
from pathlib import Path
from typing import Dict, Any
import json, shutil, csv

def new_session_dir() -> str:
    root = Path("runs"); root.mkdir(exist_ok=True)
    session = root / __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (session / "captures").mkdir(parents=True, exist_ok=True)
    (session / "diffs").mkdir(parents=True, exist_ok=True)
    return str(session)

def save_config(session_dir: str, cfg: Dict[str, Any]) -> None:
    with open(Path(session_dir)/"config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def append_index(session_dir: str, row: Dict[str, Any]) -> None:
    idx = Path(session_dir)/"index.csv"
    new = not idx.exists()
    with open(idx, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts","capture","aligned","ssim","diff_ratio","boxes"])
        if new: w.writeheader()
        w.writerow(row)

def make_zip(session_dir: str) -> str:
    base = Path(session_dir).resolve()
    out = base.parent / (base.name + ".zip")
    if out.exists(): out.unlink()
    shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=str(base))
    return str(out)