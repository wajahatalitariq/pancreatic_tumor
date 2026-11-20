# -*- coding: utf-8 -*-
"""
extract_features.py
Unified Stage 1-2 preprocessing and frozen backbone embedding extraction
matching Section 2.2 of the manuscript:
  - Resizes [0, 255] PNG axial CT slices to 256x256 (area interpolation)
  - 5x5 Gaussian smoothing (sigma = 0) + CLAHE (clipLimit=2.0, tileGridSize=8x8)
  - Harris corners (blockSize=2, ksize=3, k=0.04, R > 0.02*Rmax, d_nms=20 px, max 150)
  - Planar Delaunay triangulation rendered on 0.40x scaled CLAHE slices
Generates Swin-V2-Tiny embeddings for all five Stage 1 control variants (raw, clahe,
clahe_dim, rand_mesh, harris_mesh) plus ResNet-18 and ResNet-50 on harris_mesh.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.models as models
from scipy.spatial import Delaunay

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "CLEAN_DATASET"
CACHE_DIR = DATA_DIR / "cached_features_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_stage1_variants(img_path, clahe_obj, rng_rand):
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Missing CT slice: {img_path}")
    raw256 = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(raw256, (5, 5), sigmaX=0)
    eq = clahe_obj.apply(blur)

    resp = cv2.dilate(cv2.cornerHarris(np.float32(eq), 2, 3, 0.04), None)
    r_max = float(resp.max())
    ys, xs = np.where(resp > 0.02 * r_max) if r_max > 0 else (np.array([]), np.array([]))
    kept = []
    if len(xs) > 0:
        order = np.argsort(-resp[ys, xs])
        for x, y in zip(xs[order], ys[order]):
            if all((x - kx) ** 2 + (y - ky) ** 2 >= 20 ** 2 for kx, ky in kept):
                kept.append((int(x), int(y)))
                if len(kept) >= 150:
                    break
    pts_harris = np.array(kept, dtype=np.int32)

    def draw_mesh(base_gray, pts_arr):
        canvas = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.40
        if len(pts_arr) >= 3:
            for tri in Delaunay(pts_arr).simplices:
                for a, b in [(0, 1), (1, 2), (0, 2)]:
                    cv2.line(canvas, tuple(pts_arr[tri[a]]), tuple(pts_arr[tri[b]]), (0, 165, 255), 1, cv2.LINE_AA)
        for x, y in pts_arr:
            cv2.circle(canvas, (int(x), int(y)), 2, (255, 110, 20), -1, cv2.LINE_AA)
        return cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    n_pts = max(len(pts_harris), 3)
    pts_rand = np.column_stack([
        rng_rand.integers(8, 248, size=n_pts),
        rng_rand.integers(8, 248, size=n_pts),
    ]).astype(np.int32)

    return {
        "raw": cv2.cvtColor(raw256, cv2.COLOR_GRAY2RGB),
        "clahe": cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB),
        "clahe_dim": cv2.cvtColor(np.clip(eq.astype(np.float32) * 0.40, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
        "rand_mesh": draw_mesh(eq, pts_rand),
        "harris_mesh": draw_mesh(eq, pts_harris),
    }


def to_tensor_batch(rgb_list):
    arr = np.stack([(im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD for im in rgb_list], axis=0)
    return torch.from_numpy(arr.transpose(0, 3, 1, 2)).float()


def main():
    t0 = time.time()
    df_m = pd.read_csv(DATA_DIR / "manifest.csv")
    paths = [DATA_DIR / f for f in df_m["file"]]
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    rng_v = np.random.default_rng(SEED)

    variants_imgs = {k: [] for k in ["raw", "clahe", "clahe_dim", "rand_mesh", "harris_mesh"]}
    for p in paths:
        out_v = make_stage1_variants(p, clahe_obj, rng_v)
        for k in variants_imgs:
            variants_imgs[k].append(out_v[k])

    swin = timm.create_model("swinv2_tiny_window8_256", pretrained=True, num_classes=0).to(DEVICE).eval()
    r18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
    r18.fc = nn.Identity(); r18.eval()
    r50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
    r50.fc = nn.Identity(); r50.eval()

    def run_backbone(model, imgs_list, bs=32):
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs_list), bs):
                feats.append(model(to_tensor_batch(imgs_list[i:i+bs]).to(DEVICE)).cpu().numpy())
        return np.vstack(feats).astype(np.float32)

    for vname in ["raw", "clahe", "clahe_dim", "rand_mesh", "harris_mesh"]:
        arr = run_backbone(swin, variants_imgs[vname])
        np.save(CACHE_DIR / f"swin_{vname}.npy", arr)
        print(f"Saved swin_{vname}.npy {arr.shape}")

    np.save(CACHE_DIR / "res18_harris_mesh.npy", run_backbone(r18, variants_imgs["harris_mesh"]))
    np.save(CACHE_DIR / "res50_harris_mesh.npy", run_backbone(r50, variants_imgs["harris_mesh"]))
    np.save(CACHE_DIR / "harris_mesh_rgb256.npy", np.stack(variants_imgs["harris_mesh"], axis=0))
    print(f"Feature extraction completed in {time.time() - t0:.1f}s -> {CACHE_DIR}")


if __name__ == "__main__":
    main()
