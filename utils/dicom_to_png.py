'''
# 1) Install deps
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install pydicom pillow pandas numpy tqdm scikit-image

# 2) Put your 4 CSVs where the script can read them, then run:
python convert_vindr_to_yolo_parallel.py \
  --base /home/ec2-user/vindr \
  --out  /home/ec2-user/vindr_converted \
  --ann-train /home/ec2-user/vindr/annotations_train.csv \
  --ann-test  /home/ec2-user/vindr/annotations_test.csv \
  --labels-train /home/ec2-user/vindr/image_labels_train.csv \
  --labels-test  /home/ec2-user/vindr/image_labels_test.csv \
  --workers $(nproc)

'''




#!/usr/bin/env python3
import argparse, os, sys, json, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.exposure import rescale_intensity
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# -------- CSV column aliases --------
COL_IMAGE_ID   = ["image_id", "image", "filename", "file_name"]
COL_CLASS_NAME = ["class_name", "category", "label", "finding", "class"]
COL_XMIN       = ["xmin", "x_min", "x1", "left"]
COL_YMIN       = ["ymin", "y_min", "y1", "top"]
COL_XMAX       = ["xmax", "x_max", "x2", "right"]
COL_YMAX       = ["ymax", "y_max", "y2", "bottom"]

LABELS_IMAGE_ID = ["image_id", "image", "filename", "file_name"]
LABELS_CLASS    = ["class_name", "category", "label", "finding", "class"]

def first_existing(cols, df): 
    for c in cols:
        if c in df.columns: return c
    return None

def standardize_ann_df(df: pd.DataFrame) -> pd.DataFrame:
    m = {
        "image_id":   first_existing(COL_IMAGE_ID, df),
        "class_name": first_existing(COL_CLASS_NAME, df),
        "xmin":       first_existing(COL_XMIN, df),
        "ymin":       first_existing(COL_YMIN, df),
        "xmax":       first_existing(COL_XMAX, df),
        "ymax":       first_existing(COL_YMAX, df),
    }
    missing = [k for k,v in m.items() if v is None]
    if missing:
        raise ValueError(f"Annotations CSV missing columns: {missing}")
    out = pd.DataFrame({k: df[v] for k,v in m.items()})
    return out

def build_class_list(ann_train, labels_train, ann_test, labels_test) -> List[str]:
    def collect_from_labels(df):
        if df is None: return []
        col = first_existing(LABELS_CLASS, df)
        if col is None: return []
        vals = []
        for v in df[col].dropna().astype(str).values:
            parts = [p.strip() for p in str(v).replace("|",",").replace(";",",").split(",") if p.strip()]
            vals.extend(parts if parts else [v])
        return vals
    classes = []
    classes += collect_from_labels(labels_train)
    classes += collect_from_labels(labels_test)
    if not classes:
        a = [] if ann_train is None else ann_train["class_name"].dropna().tolist()
        b = [] if ann_test  is None else ann_test["class_name"].dropna().tolist()
        classes = a + b
    seen, ordered = set(), []
    for c in classes:
        if c not in seen:
            seen.add(c); ordered.append(c)
    if not ordered:
        raise ValueError("Could not infer class list—check labels/annotations CSVs.")
    return ordered

def dicom_to_uint8_rgb(ds: pydicom.Dataset) -> np.ndarray:
    arr = apply_voi_lut(ds.pixel_array, ds)
    arr = arr.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2").upper() == "MONOCHROME1":
        arr = np.max(arr) - arr
    lo, hi = np.percentile(arr, (2, 98))
    if hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    arr = np.clip(arr, lo, hi)
    arr = rescale_intensity(arr, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)

def write_yolo_labels(anns_rows: List[Tuple[str,float,float,float,float]], class_to_id: Dict[str,int],
                      img_w: int, img_h: int, out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cname, x1, y1, x2, y2 in anns_rows:
        if cname not in class_to_id: 
            continue
        cx = ((x1 + x2)/2.0) / img_w
        cy = ((y1 + y2)/2.0) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        cx, cy, bw, bh = [min(0.999, max(0.0, v)) for v in (cx,cy,bw,bh)]
        lines.append(f"{class_to_id[cname]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))

def save_png(img: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)

def gather_dicom_files(root: Path) -> List[Path]:
    return list(root.rglob("*.dicom"))

# ---------- worker ----------
def worker_convert(args_tuple):
    dcm_path, src_root, out_root, split, anns_by_stem, class_to_id = args_tuple
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
        rgb = dicom_to_uint8_rgb(ds)
        h, w = rgb.shape[:2]

        # mirror relative path under images/<split>/*.png
        rel = dcm_path.relative_to(src_root)
        out_img = (out_root / "images" / split / rel).with_suffix(".png")
        save_png(rgb, out_img)

        # labels path mirrors images path name
        out_lbl = (out_root / "labels" / split / rel).with_suffix(".txt")

        stem = dcm_path.stem
        rows = anns_by_stem.get(stem, [])
        if rows:
            write_yolo_labels(rows, class_to_id, w, h, out_lbl)
        else:
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            open(out_lbl, "w").close()
        return True, None
    except Exception as e:
        return False, f"{dcm_path}: {e}"

def build_ann_index(ann_df: Optional[pd.DataFrame]) -> Dict[str, List[Tuple[str,float,float,float,float]]]:
    """Return stem -> list[(class, x1,y1,x2,y2)] (picklable, small)."""
    if ann_df is None or ann_df.empty:
        return {}
    df = ann_df.copy()
    df["stem"] = df["image_id"].astype(str).map(lambda s: Path(s).stem)
    out: Dict[str, List[Tuple[str,float,float,float,float]]] = {}
    for _, r in df.iterrows():
        lst = out.setdefault(r["stem"], [])
        lst.append((str(r["class_name"]), float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])))
    return out

def write_data_yaml(out_root: Path, class_list: List[str]):
    names_block = "\n  ".join(f"{i}: {n}" for i,n in enumerate(class_list))
    content = f"""# Auto-generated
path: {out_root.as_posix()}
train: images/train
val: images/test
test: images/test
names:
  {names_block}
"""
    (out_root/"data.yaml").write_text(content)

def process_split_parallel(split: str, base: Path, out_root: Path,
                           ann_df: Optional[pd.DataFrame],
                           class_to_id: Dict[str,int],
                           workers: int) -> Tuple[int,int,int]:
    src_root = base / split
    if not src_root.exists():
        print(f"[{split}] missing: {src_root}, skipping", file=sys.stderr)
        return 0,0,0
    files = gather_dicom_files(src_root)
    ann_idx = build_ann_index(ann_df)

    ok, fail = 0, 0
    tasks = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for dcm in files:
            tasks.append(ex.submit(worker_convert, (dcm, src_root, out_root, split, ann_idx, class_to_id)))
        for fut in tqdm(as_completed(tasks), total=len(tasks), desc=f"[{split}] Converting in parallel"):
            success, err = fut.result()
            if success: ok += 1
            else:
                fail += 1
                print(f"[{split}] ERROR {err}", file=sys.stderr)
    # images written = ok; labels written ~ images with any ann
    lbl_count = sum(1 for stem, rows in ann_idx.items() if rows)
    return ok, lbl_count, fail

def main():
    ap = argparse.ArgumentParser("Fast VinDr-CXR DICOM → PNG (YOLO) with parallel processing")
    ap.add_argument("--base", required=True, help="Base path containing train/ and test/ with DICOMs")
    ap.add_argument("--out",  required=True, help="Output root path")
    ap.add_argument("--ann-train", required=True, help="CSV with TRAIN bboxes")
    ap.add_argument("--ann-test",  required=True, help="CSV with TEST bboxes")
    ap.add_argument("--labels-train", help="CSV with TRAIN image-level labels (optional)")
    ap.add_argument("--labels-test",  help="CSV with TEST image-level labels (optional)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()-1), help="Parallel worker processes")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ann_train = standardize_ann_df(pd.read_csv(args.ann_train)) if args.ann_train else None
    ann_test  = standardize_ann_df(pd.read_csv(args.ann_test))  if args.ann_test  else None
    labels_train = pd.read_csv(args.labels_train) if args.labels_train else None
    labels_test  = pd.read_csv(args.labels_test)  if args.labels_test  else None

    classes = build_class_list(ann_train, labels_train, ann_test, labels_test)
    class_to_id = {c:i for i,c in enumerate(classes)}
    print("Classes:", classes)

    img_t, lbl_t, fail_t = process_split_parallel("train", base, out_root, ann_train, class_to_id, args.workers)
    img_s, lbl_s, fail_s = process_split_parallel("test",  base, out_root, ann_test,  class_to_id, args.workers)

    write_data_yaml(out_root, classes)
    summary = {
        "output_root": str(out_root),
        "train": {"images_written": img_t, "estimated_labels_with_boxes": lbl_t, "failed": fail_t},
        "test":  {"images_written": img_s, "estimated_labels_with_boxes": lbl_s, "failed": fail_s},
        "classes": classes,
        "workers": args.workers
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

