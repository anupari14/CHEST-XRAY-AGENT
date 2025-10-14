# models/yolo.py
from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import os, cv2, torch

from ultralytics import YOLO


class YOLOInferencer:
    def __init__(self, weights: str | Path, device: str | int | None = None):
        self.weights = str(weights)
        if device is None:
            device = 0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
        self.device = device
        self.model = YOLO(self.weights)

    def infer_probs_and_overlays(
        self,
        image_path: str | Path,
        out_dir: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        topk: int = 3,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Returns:
          findings: [{'label': str, 'prob': float, 'present': bool}, ...]  (all classes, sorted desc)
          cams:     [{'rank': int, 'label': str, 'score': float, 'cam_png': '/abs/path.png'}, ...] (top-K)
        Also writes overlay PNGs (per class) into out_dir.
        """
        image_path = Path(image_path)
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

        res = self.model.predict(
            source=str(image_path),
            device=self.device,
            conf=conf,
            iou=iou,
            imgsz=512,
            stream=False,
            save=False,
            verbose=False,
        )[0]

        names = res.names                               # dict: id -> class_name
        num_classes = len(names)

        # per-class max confidence + keep boxes by class
        max_conf = {i: 0.0 for i in range(num_classes)}
        boxes_by_cls = {i: [] for i in range(num_classes)}
        if res.boxes is not None and len(res.boxes) > 0:
            for x1, y1, x2, y2, cf, cls_id in res.boxes.data.cpu().numpy():
                i = int(cls_id); cf = float(cf)
                if cf > max_conf[i]:
                    max_conf[i] = cf
                boxes_by_cls[i].append((float(x1), float(y1), float(x2), float(y2), cf))

        # findings (all classes)
        findings = [{
            "label": names[i] if names[i] is not None else "",
            "prob": float(max_conf[i]),
            "present": (max_conf[i] >= conf),
        } for i in range(num_classes)]
        findings.sort(key=lambda d: d["prob"], reverse=True)

        # top-K overlays by per-class max confidence
        top_cls = [i for i in range(num_classes) if max_conf[i] > 0.0]
        top_cls.sort(key=lambda i: max_conf[i], reverse=True)
        top_cls = top_cls[:max(1, int(topk))]

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        cams: List[dict] = []
        for rank, cls_id in enumerate(top_cls, start=1):
            label = names[cls_id] if names[cls_id] is not None else ""
            score = float(max_conf[cls_id])

            canvas = img_bgr.copy()
            for (x1, y1, x2, y2, cf) in boxes_by_cls[cls_id]:
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                txt = f"{label}: {cf:.3f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(canvas, (int(x1), int(y1) - th - 8), (int(x1) + tw + 6, int(y1)), (0, 0, 0), -1)
                cv2.putText(canvas, txt, (int(x1) + 3, int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            safe_label = (label or "").replace(" ", "_").replace("/", "_")
            out_png = out_dir / f"cam_{rank}_{safe_label}.png"
            cv2.imwrite(str(out_png), canvas)

            cams.append({
                "rank": rank,
                "label": label,
                "score": score,
                "cam_png": str(out_png.resolve()),
            })

        return findings, cams
