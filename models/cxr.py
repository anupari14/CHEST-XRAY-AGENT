# models/cxr.py  (PATCHED)
import os, json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torchvision
import torchxrayvision as xrv
from torchcam.methods import GradCAM

try:
    import pydicom
except Exception:
    pydicom = None

DEFAULT_TOPK = 3
DEFAULT_SAVE_DIR = "cam_outputs"
ALPHA = 0.45
BBOX_THR = 0.6

def _to_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CXRInferencer:
    def __init__(self, weights: str = "densenet121-res224-chex", device: Optional[str] = None):
        self.device = torch.device(device) if device else _to_device()
        self.model = xrv.models.DenseNet(weights=weights).to(self.device).eval()
        self.labels = self.model.pathologies
        self._last_conv = self.model.features.denseblock4.denselayer16.conv2
        self._tfm = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])

    # ---------- UTIL: aggressively clear any lingering forward/backward hooks ----------
    def _clear_torchcam_hooks(self) -> None:
        for m in self.model.modules():
            if hasattr(m, "_forward_hooks"):
                m._forward_hooks.clear()
            if hasattr(m, "_backward_hooks"):
                m._backward_hooks.clear()
            if hasattr(m, "_forward_pre_hooks"):
                m._forward_pre_hooks.clear()

    # ---------- IO / preprocess ----------
    def _load_image_to_224(self, path: str) -> torch.Tensor:
        if path.lower().endswith(".dcm"):
            if not pydicom:
                raise RuntimeError("pydicom is required for DICOM files")
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
            mn, mx = float(np.min(arr)), float(np.max(arr))
            arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr, dtype=np.float32)
        else:
            pil = Image.open(path).convert("L")
            arr = np.array(pil, dtype=np.float32)
            arr = xrv.datasets.normalize(arr, 255.0)

        if arr.ndim == 2:
            arr = arr[None, :, :]
        ten = self._tfm(arr)                      # (1,224,224) in [0,1]
        ten = torch.as_tensor(ten).float()[None]  # (1,1,224,224)
        return ten.to(self.device)

    # ---------- Predictions (no grads) ----------
    @torch.no_grad()
    def predict(self, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        # ensure no stale CAM hooks are attached before plain inference
        self._clear_torchcam_hooks()

        x = self._load_image_to_224(image_path)
        logits = self.model(x)  # (1, num_labels)
        probs = torch.sigmoid(logits)[0].cpu().numpy().tolist()

        findings = []
        for i, lbl in enumerate(self.labels):
            p = float(probs[i])
            findings.append({"label": lbl, "prob": p, "present": p >= threshold})
        findings.sort(key=lambda d: d["prob"], reverse=True)

        return {
            "model": "torchxrayvision DenseNet121 (CheXpert)",
            "labels": self.labels,
            "threshold": threshold,
            "findings": findings,
        }

    # ---------- CAM helpers ----------
    @staticmethod
    def _tensor_to_base_rgb(x: torch.Tensor) -> Image.Image:
        g = x[0, 0].detach().cpu().numpy()
        g = np.clip(g, 0.0, 1.0)
        img = (np.stack([g, g, g], axis=-1) * 255.0).astype(np.uint8)
        return Image.fromarray(img)

    @staticmethod
    def _resize_cam_to_input(cam_np: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
        H, W = hw
        t = torch.from_numpy(cam_np).float()[None, None, ...]
        t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        t = F.avg_pool2d(t, 5, stride=1, padding=2)
        cam = t.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    @staticmethod
    def _weak_bbox(cam_norm: np.ndarray, thr: float = BBOX_THR):
        ys, xs = np.where(cam_norm >= thr)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _overlay_red(base_pil: Image.Image, cam_norm: np.ndarray, alpha: float,
                     label: Optional[str], score: Optional[float],
                     draw_bbox: bool, out_png: str) -> None:
        base = np.asarray(base_pil).astype(np.float32)
        heat = np.zeros_like(base)
        heat[..., 0] = cam_norm * 255.0
        blended = (alpha * heat + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)
        out = Image.fromarray(blended)

        if draw_bbox:
            bb = CXRInferencer._weak_bbox(cam_norm)
            if bb is not None:
                ImageDraw.Draw(out).rectangle(bb, outline="red", width=3)

        if label is not None and score is not None:
            draw = ImageDraw.Draw(out)
            text = f"{label}: {score:.3f}"
            w = 8 * len(text) + 12
            draw.rectangle([5, 5, 5 + w, 28], fill="black")
            draw.text((10, 8), text, fill="white")

        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        out.save(out_png)

    # ---------- Top-K Grad-CAM (with guaranteed hook cleanup) ----------
    def topk_cam(self, image_path: str, topk: int = DEFAULT_TOPK, save_dir: str = DEFAULT_SAVE_DIR) -> Dict[str, Any]:
        os.makedirs(save_dir, exist_ok=True)

        # IMPORTANT: allow grads for CAM; do not use no_grad here
        x = self._load_image_to_224(image_path)
        base = self._tensor_to_base_rgb(x)
        H, W = x.shape[-2], x.shape[-1]

        cam_extractor = GradCAM(self.model, target_layer=self._last_conv)
        try:
            logits = self.model(x)  # keep graph for GradCAM
            scores = logits[0].detach().cpu().numpy()
            idx_sorted = np.argsort(-scores)
            topk_idx = idx_sorted[:max(1, int(topk))]

            results: List[Dict[str, Any]] = []
            for i, idx in enumerate(topk_idx):
                lab = self.labels[int(idx)]
                sc = float(logits[0, int(idx)].item())
                retain = (i < len(topk_idx) - 1)
                cams = cam_extractor(class_idx=int(idx), scores=logits, retain_graph=retain)
                cam = cams[-1].squeeze().detach().cpu().numpy()
                cam = self._resize_cam_to_input(cam, (H, W))

                safe_lab = lab.replace(" ", "_").replace("/", "_")
                out_png = os.path.join(save_dir, f"cam_{i+1}_{safe_lab}.png")
                self._overlay_red(base, cam, alpha=ALPHA, label=lab, score=sc, draw_bbox=True, out_png=out_png)

                results.append({"rank": i + 1, "label": lab, "score": sc, "cam_png": out_png})

            # Save JSON artifacts
            preds_path = os.path.join(save_dir, "predictions.json")
            topk_path = os.path.join(save_dir, "topk_cam_outputs.json")
            with open(preds_path, "w") as f:
                json.dump({lab: float(logits[0, j].item()) for j, lab in enumerate(self.labels)}, f, indent=2)
            with open(topk_path, "w") as f:
                json.dump(results, f, indent=2)

            return {
                "image": image_path,
                "save_dir": save_dir,
                "predictions_json": preds_path,
                "topk_cam_json": topk_path,
                "results": results,
            }
        finally:
            # ALWAYS detach hooks so future no_grad inference won't crash
            try:
                cam_extractor.remove_hooks()
            except Exception:
                pass
