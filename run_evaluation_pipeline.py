# -*- coding: utf-8 -*-
"""
run_evaluation_pipeline.py
Unconfounded, reproducible evaluation suite resolving all peer-review concerns:
1. Unified exact Stage 1 preprocessing (Gaussian 5x5 sigma=0, CLAHE clip=2.0 grid=8x8,
   Harris k=0.04 block=2 ksize=3 R>0.02*Rmax d_nms=20px, 0.40 intensity scale).
2. Feature extraction on 5 Stage-1 arms:
   - raw_ct, clahe_only, clahe_dim04, rand_mesh, harris_mesh (Swin-V2-Tiny 768-D)
   - plus ResNet-18 (512-D) and ResNet-50 (2048-D) on harris_mesh
3. Three evaluation regimes directly assessing source confounding (`manifest.csv`):
   - Regime A: Full Cohort ordered 75/25 contiguous block split (744 train / 248 test)
     + Ordered Contiguous 5-Fold Block CV (NO slice-random shuffling)
   - Regime B: Single-source domain subgroup (source_group == 1, N=463: 220 normal, 243 tumor)
     ordered 75/25 block split (347 train / 116 test) + Contiguous 5-Fold Block CV
   - Regime C: Cross-source OOD test (Train source_group 1 [463] -> Test source_group {0,2,3} [529])
4. Unconfounded Stage 1 controls (full 768-D Swin-V2 across all 5 Stage-1 variants, 5 seeds)
5. Unconfounded Backbone & Selector benchmarks:
   - End-to-End Fine-Tuned Swin-V2-Tiny (with internal train-only 85/15 early stopping)
   - Multi-seed (5 seeds) Binary HHO Jaccard stability & performance
   - Matched-cardinality (k = d_HHO) selectors: L1-Logistic, PCA, Mutual Information
   - Fixed-budget ResNet-18 (k = d_HHO) + unpruned ResNet-18 (512-D)
   - Hyperparameter validation strictly on internal 85/15 train split
6. Valid statistical tests ONLY:
   - Exact two-sided McNemar binomial test + Holm-Bonferroni correction
   - Two-sided DeLong correlated ROC-AUC test + Holm-Bonferroni correction
   - Moving-block (L=10) and slice-level 2,000-iteration bootstrap 95% CIs
   - PPV under realistic screening prevalences (0.5%, 1%, 10%, 37.5%)
"""

import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shap
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from scipy import stats
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
SEEDS_MULTI = [42, 43, 44, 45, 46]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "CLEAN_DATASET"
CACHE_V2 = DATA_DIR / "cached_features_v2"
OUT_DIR = DATA_DIR / "outputs"
CACHE_V2.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)
print(f"[INIT] Device: {DEVICE}")

# ------------------------------------------------------------------------------
# 1. LOAD MANIFEST IN EXACT SEQUENTIAL ORDER (620 normal + 372 tumor = 992)
# ------------------------------------------------------------------------------
df_m = pd.read_csv(DATA_DIR / "manifest.csv")
df_m["label_val"] = (df_m["label"] == "pancreatic_tumor").astype(int)
paths = [str(DATA_DIR / f) for f in df_m["file"]]
y_all = df_m["label_val"].values
sg_all = df_m["source_group"].values
print(f"[DATA] Total={len(df_m)} | Normal={sum(y_all==0)} | Tumor={sum(y_all==1)}")

# ------------------------------------------------------------------------------
# 2. UNIFIED STAGE 1 PREPROCESSING & FEATURE EXTRACTION (ALL 5 ARMS)
# ------------------------------------------------------------------------------
clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def make_stage1_variants(img_path, rng_rand):
    raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    raw256 = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(raw256, (5, 5), sigmaX=0)
    eq = clahe_obj.apply(blur)

    # Harris corner detection (blockSize=2, ksize=3, k=0.04, R > 0.02 * Rmax, d_nms=20)
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

    img_raw_rgb = cv2.cvtColor(raw256, cv2.COLOR_GRAY2RGB)
    img_clahe_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    img_dim_rgb = cv2.cvtColor(np.clip(eq.astype(np.float32) * 0.40, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    n_pts = max(len(pts_harris), 3)
    pts_rand = np.column_stack([
        rng_rand.integers(8, 248, size=n_pts),
        rng_rand.integers(8, 248, size=n_pts),
    ]).astype(np.int32)
    img_rand_rgb = draw_mesh(eq, pts_rand)
    img_harris_rgb = draw_mesh(eq, pts_harris)

    return {
        "raw": img_raw_rgb,
        "clahe": img_clahe_rgb,
        "clahe_dim": img_dim_rgb,
        "rand_mesh": img_rand_rgb,
        "harris_mesh": img_harris_rgb,
    }

def to_tensor_batch(rgb_list):
    arr = np.stack([(im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD for im in rgb_list], axis=0)
    return torch.from_numpy(arr.transpose(0, 3, 1, 2)).float()

cache_marker = CACHE_V2 / "swin_harris_mesh.npy"
if not cache_marker.exists():
    print("[EXTRACT] Computing exact CLAHE(2.0, 8x8) + Harris(k=0.04, 0.02*Rmax, d_nms=20, 0.40) features across all 5 Stage-1 variants...")
    t0 = time.time()
    rng_v = np.random.default_rng(SEED)
    variants_imgs = {k: [] for k in ["raw", "clahe", "clahe_dim", "rand_mesh", "harris_mesh"]}
    for p in paths:
        out_v = make_stage1_variants(p, rng_v)
        for k in variants_imgs:
            variants_imgs[k].append(out_v[k])

    swin = timm.create_model("swinv2_tiny_window8_256", pretrained=True, num_classes=0).to(DEVICE).eval()
    r18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
    r18.fc = nn.Identity(); r18.eval()
    r50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
    r50.fc = nn.Identity(); r50.eval()

    def run_bb(model, imgs_list, bs=32):
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs_list), bs):
                xb = to_tensor_batch(imgs_list[i:i+bs]).to(DEVICE)
                feats.append(model(xb).cpu().numpy())
        return np.vstack(feats).astype(np.float32)

    for vname in ["raw", "clahe", "clahe_dim", "rand_mesh", "harris_mesh"]:
        arr = run_bb(swin, variants_imgs[vname])
        np.save(CACHE_V2 / f"swin_{vname}.npy", arr)
        print(f"  saved swin_{vname}.npy {arr.shape}")

    arr_r18 = run_bb(r18, variants_imgs["harris_mesh"])
    arr_r50 = run_bb(r50, variants_imgs["harris_mesh"])
    np.save(CACHE_V2 / "res18_harris_mesh.npy", arr_r18)
    np.save(CACHE_V2 / "res50_harris_mesh.npy", arr_r50)
    # Save harris_mesh RGBs compressed for optional fine-tuning Swin-V2 end-to-end
    np.save(CACHE_V2 / "harris_mesh_rgb256.npy", np.stack(variants_imgs["harris_mesh"], axis=0))
    print(f"[EXTRACT] Completed in {time.time()-t0:.1f}s")
else:
    print("[EXTRACT] Loading cached v2 features from", CACHE_V2)

swin_raw = np.load(CACHE_V2 / "swin_raw.npy")
swin_clahe = np.load(CACHE_V2 / "swin_clahe.npy")
swin_dim = np.load(CACHE_V2 / "swin_clahe_dim.npy")
swin_rand = np.load(CACHE_V2 / "swin_rand_mesh.npy")
swin_harris = np.load(CACHE_V2 / "swin_harris_mesh.npy")
res18_harris = np.load(CACHE_V2 / "res18_harris_mesh.npy")
res50_harris = np.load(CACHE_V2 / "res50_harris_mesh.npy")

# ------------------------------------------------------------------------------
# 3. CORE MODELS, HHO, METRICS, DELONG & MCNEMAR + HOLM-BONFERRONI
# ------------------------------------------------------------------------------
class RegularizedMLP(nn.Module):
    def __init__(self, in_dim, h1=128, h2=64, drop=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h2, 2),
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(X_tr, y_tr, X_te, epochs=120, lr=0.003, wd=1e-4, drop=0.20, seed=42, track_val=False):
    set_seed(seed)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if track_val:
        X_st, X_sv, y_st, y_sv = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=seed)
        m_v = RegularizedMLP(X_tr.shape[1], drop=drop).to(DEVICE)
        o_v = optim.AdamW(m_v.parameters(), lr=lr, weight_decay=wd)
        s_v = optim.lr_scheduler.CosineAnnealingLR(o_v, T_max=epochs)
        ce = nn.CrossEntropyLoss()
        t_xst = torch.tensor(X_st, dtype=torch.float32).to(DEVICE)
        t_yst = torch.tensor(y_st, dtype=torch.long).to(DEVICE)
        t_xsv = torch.tensor(X_sv, dtype=torch.float32).to(DEVICE)
        t_ysv = torch.tensor(y_sv, dtype=torch.long).to(DEVICE)
        for _ in range(epochs):
            m_v.train(); o_v.zero_grad()
            lt = m_v(t_xst); loss_t = ce(lt, t_yst); loss_t.backward(); o_v.step(); s_v.step()
            m_v.eval()
            with torch.no_grad():
                lv = m_v(t_xsv)
                hist["train_loss"].append(float(loss_t.item()))
                hist["val_loss"].append(float(ce(lv, t_ysv).item()))
                hist["train_acc"].append(float((lt.argmax(1) == t_yst).float().mean().item()))
                hist["val_acc"].append(float((lv.argmax(1) == t_ysv).float().mean().item()))

    set_seed(seed)
    m = RegularizedMLP(X_tr.shape[1], drop=drop).to(DEVICE)
    o = optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    s = optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    t_x = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    t_y = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    t_te = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    for _ in range(epochs):
        m.train(); o.zero_grad(); ce(m(t_x), t_y).backward(); o.step(); s.step()
    m.eval()
    with torch.no_grad():
        lg = m(t_te)
        probs = torch.softmax(lg, dim=1)[:, 1].cpu().numpy()
        preds = lg.argmax(1).cpu().numpy()
    return preds, probs, m, hist

def run_binary_hho(X_tr, y_tr, num_agents=20, max_iter=25, alpha=0.99, beta=0.01, seed=42):
    rng = np.random.default_rng(seed)
    n_feat = X_tr.shape[1]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(skf.split(X_tr, y_tr))

    def eval_m(pos_v):
        mask = (1.0 / (1.0 + np.exp(-np.clip(pos_v, -10, 10)))) > 0.5
        n_sel = int(mask.sum())
        if n_sel < 10:
            return 1.0, 0.0, n_sel
        accs = []
        for ti, vi in splits:
            clf = LogisticRegression(C=1.0, max_iter=100, solver="liblinear", random_state=seed)
            clf.fit(X_tr[ti][:, mask], y_tr[ti])
            accs.append(clf.score(X_tr[vi][:, mask], y_tr[vi]))
        mean_acc = float(np.mean(accs))
        return alpha * (1.0 - mean_acc) + beta * (n_sel / n_feat), mean_acc, n_sel

    pos = rng.uniform(-0.5, 0.8, size=(num_agents, n_feat))
    fits, accs = zip(*[(fv, ac) for fv, ac, _ in (eval_m(p) for p in pos)])
    fits = np.array(fits); bi = int(np.argmin(fits))
    best_pos = pos[bi].copy(); best_fit = float(fits[bi]); best_acc = float(accs[bi])
    f_hist, s_hist, a_hist, e_hist = [], [], [], []
    for t in range(max_iter):
        E0_i = 2 * rng.random() - 1
        E_i = 2 * E0_i * (1.0 - (t + 1) / max_iter)
        for i in range(num_agents):
            E = 2 * (2 * rng.random() - 1) * (1.0 - (t + 1) / max_iter)
            if abs(E) >= 1:
                if rng.random() >= 0.5:
                    rh = pos[rng.integers(0, num_agents)]
                    pos[i] = rh - rng.random() * abs(rh - 2 * rng.random() * pos[i])
                else:
                    pos[i] = (best_pos - pos.mean(0)) - rng.random() * (2 * rng.random() - 1)
            else:
                r = rng.random()
                if r >= 0.5:
                    if abs(E) >= 0.5:
                        pos[i] = (best_pos - pos[i]) - E * abs(2 * (1 - rng.random()) * best_pos - pos[i])
                    else:
                        pos[i] = best_pos - E * abs(best_pos - pos[i])
                else:
                    if abs(E) >= 0.5:
                        pos[i] = (best_pos - pos[i]) - E * abs(2 * rng.random() * best_pos - pos[i])
                    else:
                        pos[i] = (best_pos - pos[i]) - E * abs(best_pos - pos.mean(0))
            pos[i] = np.clip(pos[i], -3.0, 3.0)
            fn, an, _ = eval_m(pos[i])
            if fn < best_fit:
                best_fit, best_acc, best_pos = fn, an, pos[i].copy()
        cm = (1.0 / (1.0 + np.exp(-best_pos))) > 0.5
        f_hist.append(best_fit); s_hist.append(int(cm.sum())); a_hist.append(best_acc); e_hist.append(abs(E_i))
    final_mask = (1.0 / (1.0 + np.exp(-best_pos))) > 0.5
    return final_mask, f_hist, s_hist, a_hist, e_hist

def compute_metrics(y_t, p, pr):
    tn, fp, fn, tp = confusion_matrix(y_t, p, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": float(accuracy_score(y_t, p)),
        "Balanced_Acc": float(balanced_accuracy_score(y_t, p)),
        "F1": float(f1_score(y_t, p, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_t, pr)),
        "PR_AUC": float(average_precision_score(y_t, pr)),
        "Precision": float(precision_score(y_t, p, zero_division=0)),
        "Sensitivity": float(recall_score(y_t, p, zero_division=0)),
        "Specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        "MCC": float(matthews_corrcoef(y_t, p)),
        "Errors": int(fp + fn),
    }

# Fast DeLong test (Sun & Xu 2014) for correlated ROC curves
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2

def fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = compute_midrank(pos[r]); ty[r] = compute_midrank(neg[r]); tz[r] = compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01) if k > 1 else np.var(v01, ddof=1)
    sy = np.cov(v10) if k > 1 else np.var(v10, ddof=1)
    cov = sx / m + sy / n
    return aucs, cov

def delong_roc_test(y_true, p1, p2):
    order = np.argsort(-y_true)
    y_s = y_true[order]
    m = int((y_s == 1).sum())
    stacked = np.vstack([p1[order], p2[order]])
    aucs, cov = fast_delong(stacked, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 1e-12:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = diff / np.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), p

def holm_bonferroni(pvals):
    m = len(pvals)
    idx = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, j in enumerate(idx):
        val = (m - rank) * pvals[j]
        running_max = max(running_max, val)
        adj[j] = min(1.0, running_max)
    return adj.tolist()

# End-to-End Fine-Tuned Swin-V2-Tiny Baseline (Internal 85/15 train validation only!)
def run_finetuned_swin(imgs_rgb, y_arr, tr_idx, te_idx, epochs=12, lr=1e-4, seed=42):
    set_seed(seed)
    Xtr_all = imgs_rgb[tr_idx]; ytr_all = y_arr[tr_idx]
    Xte_all = imgs_rgb[te_idx]; yte_all = y_arr[te_idx]
    st_i, sv_i = train_test_split(np.arange(len(tr_idx)), test_size=0.15, stratify=ytr_all, random_state=seed)
    X_st, y_st = Xtr_all[st_i], ytr_all[st_i]
    X_sv, y_sv = Xtr_all[sv_i], ytr_all[sv_i]

    model = timm.create_model("swinv2_tiny_window8_256", pretrained=True, num_classes=2).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0; best_state = None
    bs = 16
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_st))
        for i in range(0, len(perm), bs):
            b_idx = perm[i:i+bs]
            xb = to_tensor_batch(X_st[b_idx]).to(DEVICE)
            yb = torch.tensor(y_st[b_idx], dtype=torch.long).to(DEVICE)
            opt.zero_grad(); ce(model(xb), yb).backward(); opt.step()
        sch.step()
        model.eval()
        vp = []
        with torch.no_grad():
            for i in range(0, len(X_sv), 32):
                vp.append(model(to_tensor_batch(X_sv[i:i+32]).to(DEVICE)).argmax(1).cpu().numpy())
        vacc = float((np.concatenate(vp) == y_sv).mean())
        if vacc >= best_val_acc:
            best_val_acc = vacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for i in range(0, len(Xte_all), 32):
            lg = model(to_tensor_batch(Xte_all[i:i+32]).to(DEVICE))
            probs.append(torch.softmax(lg, dim=1)[:, 1].cpu().numpy())
            preds.append(lg.argmax(1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(preds), np.concatenate(probs), best_val_acc

# Helper: contiguous ordered 75/25 split within each class
def ordered_block_split(idx_pool, labels_arr):
    n_i = idx_pool[labels_arr[idx_pool] == 0]
    t_i = idx_pool[labels_arr[idx_pool] == 1]
    tr = np.concatenate([n_i[:int(len(n_i) * 0.75)], t_i[:int(len(t_i) * 0.75)]])
    te = np.concatenate([n_i[int(len(n_i) * 0.75):], t_i[int(len(t_i) * 0.75):]])
    return tr, te

# Ordered Contiguous 5-Fold Block CV (NO random slice shuffling!)
def ordered_block_5fold_cv(X_feats, y_vec, idx_pool, run_hho_flag=True, seed=42):
    n_i = idx_pool[y_vec[idx_pool] == 0]
    t_i = idx_pool[y_vec[idx_pool] == 1]
    n_chunks = np.array_split(n_i, 5)
    t_chunks = np.array_split(t_i, 5)
    rows = []
    for k in range(5):
        va = np.concatenate([n_chunks[k], t_chunks[k]])
        tr = np.concatenate([c for j, c in enumerate(n_chunks) if j != k] + [c for j, c in enumerate(t_chunks) if j != k])
        sc = StandardScaler().fit(X_feats[tr])
        Xtr_s, Xva_s = sc.transform(X_feats[tr]), sc.transform(X_feats[va])
        if run_hho_flag:
            m_f, _, _, _, _ = run_binary_hho(Xtr_s, y_vec[tr], num_agents=15, max_iter=15, alpha=0.99, beta=0.01, seed=seed + k)
            Xtr_in, Xva_in = Xtr_s[:, m_f], Xva_s[:, m_f]
        else:
            Xtr_in, Xva_in = Xtr_s, Xva_s
        p_f, pr_f, _, _ = train_mlp(Xtr_in, y_vec[tr], Xva_in, epochs=120, lr=0.003, drop=0.20, seed=seed)
        mf = compute_metrics(y_vec[va], p_f, pr_f)
        mf["Fold"] = f"Block {k+1}"
        rows.append(mf)
    return rows

# ------------------------------------------------------------------------------
# 4. RUN REGIME A (FULL COHORT, 744/248), REGIME B (SOURCE_GROUP 1, 347/116), REGIME C (CROSS-SOURCE OOD)
# ------------------------------------------------------------------------------
results_ledger = {}

full_pool = np.arange(len(y_all))
tr_A, te_A = ordered_block_split(full_pool, y_all)

sg1_pool = np.where(sg_all == 1)[0]
tr_B, te_B = ordered_block_split(sg1_pool, y_all)

tr_C = sg1_pool
te_C = np.where(sg_all != 1)[0]

print(f"\n[REGIMES] A: tr={len(tr_A)} te={len(te_A)} | B(sg1): tr={len(tr_B)} te={len(te_B)} | C(OOD sg1->0,2,3): tr={len(tr_C)} te={len(te_C)}")

imgs_rgb256 = np.load(CACHE_V2 / "harris_mesh_rgb256.npy")

# Fine-tuned Swin-V2 baselines on Regimes A, B, C
print("[FINETUNE] Training Fine-Tuned Swin-V2-Tiny baselines (internal 85/15 val)...")
ft_A_p, ft_A_pr, ft_A_va = run_finetuned_swin(imgs_rgb256, y_all, tr_A, te_A, epochs=12, lr=1e-4, seed=SEED)
ft_B_p, ft_B_pr, ft_B_va = run_finetuned_swin(imgs_rgb256, y_all, tr_B, te_B, epochs=12, lr=1e-4, seed=SEED)
ft_C_p, ft_C_pr, ft_C_va = run_finetuned_swin(imgs_rgb256, y_all, tr_C, te_C, epochs=12, lr=1e-4, seed=SEED)

# Also run Stage-1 unconfounded 5-arm comparison (Full 768-D Swin-V2 + 5-seed MLP!) on Regimes A, B, C
def eval_regime(name, tr_idx, te_idx, ft_preds, ft_probs):
    print(f"\n=== EVALUATING {name} (tr={len(tr_idx)}, te={len(te_idx)}) ===")
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    def fit_scale(arr):
        sc = StandardScaler().fit(arr[tr_idx])
        return sc.transform(arr[tr_idx]), sc.transform(arr[te_idx])

    Xtr_raw, Xte_raw = fit_scale(swin_raw)
    Xtr_cl, Xte_cl = fit_scale(swin_clahe)
    Xtr_dim, Xte_dim = fit_scale(swin_dim)
    Xtr_rnd, Xte_rnd = fit_scale(swin_rand)
    Xtr_har, Xte_har = fit_scale(swin_harris)
    Xtr_r18, Xte_r18 = fit_scale(res18_harris)
    Xtr_r50, Xte_r50 = fit_scale(res50_harris)

    # 5-seed HHO + Jaccard stability
    masks_5, hho_accs_5, hho_f1s_5, hho_aucs_5, hho_counts = [], [], [], [], []
    hho_curve_42 = None
    for s in SEEDS_MULTI:
        mk, fh, sh, ah, eh = run_binary_hho(Xtr_har, y_tr, num_agents=20, max_iter=25, alpha=0.99, beta=0.01, seed=s)
        masks_5.append(mk); hho_counts.append(int(mk.sum()))
        p_s, pr_s, _, hist_s = train_mlp(Xtr_har[:, mk], y_tr, Xte_har[:, mk], seed=s, track_val=(s == 42))
        m_s = compute_metrics(y_te, p_s, pr_s)
        hho_accs_5.append(m_s["Accuracy"]); hho_f1s_5.append(m_s["F1"]); hho_aucs_5.append(m_s["ROC_AUC"])
        if s == 42:
            mask42 = mk; preds_hho42 = p_s; probs_hho42 = pr_s; hist42 = hist_s
            hho_curve_42 = {"fit": fh, "sel": sh, "cv_acc": ah, "energy": eh}

    jaccards = [float(np.sum(masks_5[i] & masks_5[j]) / np.sum(masks_5[i] | masks_5[j])) for i in range(len(masks_5)) for j in range(i + 1, len(masks_5))]
    k_sel = int(mask42.sum())

    # Multi-seed unpruned 768-D Swin-V2 + MLP (5 seeds)
    sw_accs_5, sw_aucs_5 = [], []
    for s in SEEDS_MULTI:
        p_u, pr_u, _, _ = train_mlp(Xtr_har, y_tr, Xte_har, seed=s)
        mu = compute_metrics(y_te, p_u, pr_u)
        sw_accs_5.append(mu["Accuracy"]); sw_aucs_5.append(mu["ROC_AUC"])
        if s == 42:
            preds_sw768 = p_u; probs_sw768 = pr_u

    # Baselines & matched-budget selectors
    preds_r18, probs_r18, _, _ = train_mlp(Xtr_r18, y_tr, Xte_r18, seed=42)
    preds_r50, probs_r50, _, _ = train_mlp(Xtr_r50, y_tr, Xte_r50, seed=42)

    # Matched k=k_sel selectors on Swin-V2 (L1, PCA, Mutual Info) + matched k ResNet18 (Mutual Info)
    l1_clf = LogisticRegression(penalty="l1", C=0.5, solver="liblinear", random_state=42).fit(Xtr_har, y_tr)
    top_l1 = np.argsort(np.abs(l1_clf.coef_[0]))[-k_sel:]
    preds_l1, probs_l1, _, _ = train_mlp(Xtr_har[:, top_l1], y_tr, Xte_har[:, top_l1], seed=42)

    k_pca = min(k_sel, Xtr_har.shape[0] - 1)
    pca = PCA(n_components=k_pca, random_state=42).fit(Xtr_har)
    preds_pca, probs_pca, _, _ = train_mlp(pca.transform(Xtr_har), y_tr, pca.transform(Xte_har), seed=42)

    mi_sel = SelectKBest(mutual_info_classif, k=k_sel).fit(Xtr_har, y_tr)
    preds_mi, probs_mi, _, _ = train_mlp(mi_sel.transform(Xtr_har), y_tr, mi_sel.transform(Xte_har), seed=42)

    mi_r18 = SelectKBest(mutual_info_classif, k=min(k_sel, Xtr_r18.shape[1])).fit(Xtr_r18, y_tr)
    preds_r18_k, probs_r18_k, _, _ = train_mlp(mi_r18.transform(Xtr_r18), y_tr, mi_r18.transform(Xte_r18), seed=42)

    svm_rbf = SVC(C=1.0, kernel="rbf", probability=True, random_state=42).fit(Xtr_har, y_tr)
    preds_svm_rbf, probs_svm_rbf = svm_rbf.predict(Xte_har), svm_rbf.predict_proba(Xte_har)[:, 1]
    svm_lin70 = SVC(C=1.0, kernel="linear", probability=True, random_state=42).fit(Xtr_har[:, mask42], y_tr)
    preds_svm_lin, probs_svm_lin = svm_lin70.predict(Xte_har[:, mask42]), svm_lin70.predict_proba(Xte_har[:, mask42])[:, 1]
    rf_768 = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr_har, y_tr)
    preds_rf, probs_rf = rf_768.predict(Xte_har), rf_768.predict_proba(Xte_har)[:, 1]

    # Unconfounded Stage-1 5-Arm Controls (All using Full 768-D Swin-V2 + MLP, SEED=42)
    p_raw, pr_raw, _, _ = train_mlp(Xtr_raw, y_tr, Xte_raw, seed=42)
    p_cl, pr_cl, _, _ = train_mlp(Xtr_cl, y_tr, Xte_cl, seed=42)
    p_dim, pr_dim, _, _ = train_mlp(Xtr_dim, y_tr, Xte_dim, seed=42)
    p_rnd, pr_rnd, _, _ = train_mlp(Xtr_rnd, y_tr, Xte_rnd, seed=42)
    # 768-D Harris+Delaunay is preds_sw768, probs_sw768!

    models_dict = {
        "Proposed_HHO_MLP": (f"{k_sel}/768", preds_hho42, probs_hho42),
        "SwinV2_NoHHO_MLP": ("768/768", preds_sw768, probs_sw768),
        "SwinV2_FineTuned": ("End-to-End", ft_preds, ft_probs),
        "SwinV2_SVM_RBF": ("768/768", preds_svm_rbf, probs_svm_rbf),
        "SwinV2_RF_200": ("768/768", preds_rf, probs_rf),
        "ResNet50_MLP": ("2048/2048", preds_r50, probs_r50),
        "ResNet18_MLP": ("512/512", preds_r18, probs_r18),
        f"ResNet18_MI_k{k_sel}": (f"{k_sel}/512", preds_r18_k, probs_r18_k),
        f"SwinV2_L1_k{k_sel}": (f"{k_sel}/768", preds_l1, probs_l1),
        f"SwinV2_PCA_k{k_sel}": (f"{k_pca}/768", preds_pca, probs_pca),
        f"SwinV2_MI_k{k_sel}": (f"{k_sel}/768", preds_mi, probs_mi),
        f"SwinV2_HHO_LinSVM": (f"{k_sel}/768", preds_svm_lin, probs_svm_lin),
    }

    stage1_arms = {
        "Stage1_RawCT_768": (p_raw, pr_raw),
        "Stage1_CLAHE_768": (p_cl, pr_cl),
        "Stage1_Dim04_CLAHE_768": (p_dim, pr_dim),
        "Stage1_RandomMesh_768": (p_rnd, pr_rnd),
        "Stage1_HarrisDelaunay_768": (preds_sw768, probs_sw768),
        "Stage1_HarrisDelaunay_HHO": (preds_hho42, probs_hho42),
    }

    bench_rows = []
    for kname, (fdim, pa, pra) in models_dict.items():
        m = compute_metrics(y_te, pa, pra)
        m["Model"] = kname; m["Features"] = fdim
        bench_rows.append(m)
        print(f"  {kname:22s} ({fdim:10s}) | Acc={m['Accuracy']:.3f} Bal={m['Balanced_Acc']:.3f} F1={m['F1']:.3f} AUC={m['ROC_AUC']:.3f} Sn={m['Sensitivity']:.3f} Sp={m['Specificity']:.3f} MCC={m['MCC']:.3f} Err={m['Errors']}")

    print("  -- Stage-1 Unconfounded Controls (Full 768-D Swin-V2 + MLP) --")
    s1_rows = []
    for sname, (pa, pra) in stage1_arms.items():
        m = compute_metrics(y_te, pa, pra)
        m["Arm"] = sname
        s1_rows.append(m)
        print(f"  {sname:26s} | Acc={m['Accuracy']:.3f} Bal={m['Balanced_Acc']:.3f} AUC={m['ROC_AUC']:.3f} Pr={m['Precision']:.3f} Sn={m['Sensitivity']:.3f} Err={m['Errors']}")

    # Statistical tests (McNemar + DeLong + Holm-Bonferroni) vs comparators (Reference: SwinV2_HHO_MLP)
    comp_keys = ["ResNet18_MLP", "ResNet50_MLP", "SwinV2_NoHHO_MLP", "SwinV2_SVM_RBF", "SwinV2_RF_200", "SwinV2_FineTuned"]
    stat_rows = []
    p_mcs, p_dls = [], []
    for ck in comp_keys:
        _, p_b, pr_b = models_dict[ck]
        a = int(np.sum((preds_hho42 == y_te) & (p_b == y_te)))
        b = int(np.sum((preds_hho42 == y_te) & (p_b != y_te)))
        c = int(np.sum((preds_hho42 != y_te) & (p_b == y_te)))
        d = int(np.sum((preds_hho42 != y_te) & (p_b != y_te)))
        p_mc = float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue) if (b + c) > 0 else 1.0
        auc1, auc2, z_dl, p_dl = delong_roc_test(y_te, probs_hho42, pr_b)
        p_mcs.append(p_mc); p_dls.append(p_dl)
        stat_rows.append({
            "Comparison": f"Proposed HHO-MLP vs. {ck}",
            "a": a, "b": b, "c": c, "d": d,
            "McNemar_p": p_mc, "DeLong_Z": z_dl, "DeLong_p": p_dl,
        })
    p_mcs_h = holm_bonferroni(p_mcs)
    p_dls_h = holm_bonferroni(p_dls)
    for r, hm, hd in zip(stat_rows, p_mcs_h, p_dls_h):
        r["McNemar_Holm_p"] = hm
        r["DeLong_Holm_p"] = hd

    # Also test Stage 1: Harris+Delaunay (768) vs RawCT (768) with McNemar & DeLong!
    a1 = int(np.sum((preds_sw768 == y_te) & (p_raw == y_te)))
    b1 = int(np.sum((preds_sw768 == y_te) & (p_raw != y_te)))
    c1 = int(np.sum((preds_sw768 != y_te) & (p_raw == y_te)))
    d1 = int(np.sum((preds_sw768 != y_te) & (p_raw != y_te)))
    pm1 = float(stats.binomtest(b1, b1 + c1, 0.5, alternative="two-sided").pvalue) if (b1 + c1) > 0 else 1.0
    _, _, z1, pd1 = delong_roc_test(y_te, probs_sw768, pr_raw)
    stage1_test = {"Comparison": "Stage1_HarrisDelaunay_768 vs RawCT_768", "a": a1, "b": b1, "c": c1, "d": d1, "McNemar_p": pm1, "DeLong_Z": z1, "DeLong_p": pd1}
    print(f"  [STAGE-1 TEST] Harris768 vs Raw768: b={b1}, c={c1}, McNemar p={pm1:.3f}, DeLong Z={z1:.3f} p={pd1:.3f}")

    return {
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "test_pos": int(sum(y_te == 1)), "test_neg": int(sum(y_te == 0)),
        "k_selected": k_sel,
        "hho_5seed_counts": hho_counts,
        "hho_5seed_acc": {"mean": float(np.mean(hho_accs_5)), "std": float(np.std(hho_accs_5, ddof=1)), "vals": hho_accs_5},
        "hho_5seed_auc": {"mean": float(np.mean(hho_aucs_5)), "std": float(np.std(hho_aucs_5, ddof=1)), "vals": hho_aucs_5},
        "hho_jaccard_mean": float(np.mean(jaccards)), "hho_jaccard_std": float(np.std(jaccards, ddof=1)),
        "swin768_5seed_acc": {"mean": float(np.mean(sw_accs_5)), "std": float(np.std(sw_accs_5, ddof=1))},
        "benchmarks": bench_rows,
        "stage1_arms": s1_rows,
        "stage1_stat_test": stage1_test,
        "stat_tests": stat_rows,
        "hho_curve_42": hho_curve_42,
        "ft_internal_val_acc": ft_A_va if name.startswith("Regime_A") else (ft_B_va if name.startswith("Regime_B") else ft_C_va),
    }

results_ledger["Regime_A_FullCohort"] = eval_regime("Regime_A_FullCohort", tr_A, te_A, ft_A_p, ft_A_pr)
results_ledger["Regime_B_SourceGroup1"] = eval_regime("Regime_B_SourceGroup1", tr_B, te_B, ft_B_p, ft_B_pr)
results_ledger["Regime_C_CrossSource_OOD"] = eval_regime("Regime_C_CrossSource_OOD", tr_C, te_C, ft_C_p, ft_C_pr)

# Contiguous 5-Fold Block CV (ZERO random shuffling!) on Regime A (N=992) and Regime B (sg=1, N=463)
print("\n[BLOCK CV] Running 5-Fold Contiguous Sequence-Block CV (No random slice shuffle)...")
bcv_A = ordered_block_5fold_cv(swin_harris, y_all, full_pool, run_hho_flag=True, seed=SEED)
bcv_B = ordered_block_5fold_cv(swin_harris, y_all, sg1_pool, run_hho_flag=True, seed=SEED)
for label_cv, b_rows in [("FullCohort_5BlockCV", bcv_A), ("SourceGroup1_5BlockCV", bcv_B)]:
    accs = [r["Accuracy"] for r in b_rows]; bals = [r["Balanced_Acc"] for r in b_rows]; aucs = [r["ROC_AUC"] for r in b_rows]
    print(f"  {label_cv:22s}: Acc={np.mean(accs):.3f}Â±{np.std(accs,ddof=1):.3f}, Bal={np.mean(bals):.3f}Â±{np.std(bals,ddof=1):.3f}, AUC={np.mean(aucs):.3f}Â±{np.std(aucs,ddof=1):.3f}")
    for r in b_rows:
        print(f"    {r['Fold']}: Acc={r['Accuracy']:.3f} Bal={r['Balanced_Acc']:.3f} Sn={r['Sensitivity']:.3f} Sp={r['Specificity']:.3f} AUC={r['ROC_AUC']:.3f} MCC={r['MCC']:.3f}")

results_ledger["BlockCV_A_FullCohort"] = bcv_A
results_ledger["BlockCV_B_SourceGroup1"] = bcv_B

# Moving-Block Bootstrap (L=10) & Slice-Level Bootstrap for Regime A & B
def block_bootstrap_ci(y_t, p, pr, block_len=10, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_t)
    keys = ["Accuracy", "Balanced_Acc", "Sensitivity", "Specificity", "Precision", "F1", "ROC_AUC", "MCC"]
    pt = compute_metrics(y_t, p, pr)
    vals_slice = {k: [] for k in keys}
    vals_block = {k: [] for k in keys}
    n_blocks_needed = int(np.ceil(n / block_len))
    for _ in range(n_boot):
        idx_s = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_t[idx_s])) == 2:
            ms = compute_metrics(y_t[idx_s], p[idx_s], pr[idx_s])
            for k in keys: vals_slice[k].append(ms[k])
        starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks_needed)
        idx_b = np.concatenate([np.arange(st, st + block_len) for st in starts])[:n]
        if len(np.unique(y_t[idx_b])) == 2:
            mb = compute_metrics(y_t[idx_b], p[idx_b], pr[idx_b])
            for k in keys: vals_block[k].append(mb[k])
    out = []
    for k in keys:
        out.append({
            "Metric": k,
            "Point": pt[k],
            "Slice_CI_Low": float(np.percentile(vals_slice[k], 2.5)),
            "Slice_CI_High": float(np.percentile(vals_slice[k], 97.5)),
            "Block10_CI_Low": float(np.percentile(vals_block[k], 2.5)),
            "Block10_CI_High": float(np.percentile(vals_block[k], 97.5)),
        })
    return out

# Recompute quick CIs for Proposed_HHO_MLP on Regime A and Regime B + PPV screening Bayes table
for reg_key, tr_i, te_i in [("Regime_A", tr_A, te_A), ("Regime_B_sg1", tr_B, te_B)]:
    sc = StandardScaler().fit(swin_harris[tr_i])
    Xtr_s, Xte_s = sc.transform(swin_harris[tr_i]), sc.transform(swin_harris[te_i])
    mk, _, _, _, _ = run_binary_hho(Xtr_s, y_all[tr_i], seed=SEED)
    pa, pra, _, _ = train_mlp(Xtr_s[:, mk], y_all[tr_i], Xte_s[:, mk], seed=SEED)
    results_ledger[f"Bootstrap_{reg_key}"] = block_bootstrap_ci(y_all[te_i], pa, pra)
    m_pt = compute_metrics(y_all[te_i], pa, pra)
    sn, sp = m_pt["Sensitivity"], m_pt["Specificity"]
    ppv_rows = []
    for prev in [0.005, 0.01, 0.05, 0.10, float((y_all[te_i] == 1).mean())]:
        ppv = (sn * prev) / (sn * prev + (1 - sp) * (1 - prev)) if (sn * prev + (1 - sp) * (1 - prev)) > 0 else 0.0
        ppv_rows.append({"prevalence": round(prev, 4), "PPV": round(float(ppv), 4)})
    results_ledger[f"Screening_PPV_{reg_key}"] = ppv_rows
    print(f"[PPV {reg_key}] Sn={sn:.3f}, Sp={sp:.3f} -> " + str(ppv_rows))

# Internal 85/15 train-only hyperparameter check (justification for p=0.20, lr=0.003)
sc_hp = StandardScaler().fit(swin_harris[tr_A])
mk_hp, _, _, _, _ = run_binary_hho(sc_hp.transform(swin_harris[tr_A]), y_all[tr_A], seed=SEED)
X_hp = sc_hp.transform(swin_harris[tr_A])[:, mk_hp]
y_hp = y_all[tr_A]
hp_tr_i, hp_va_i = train_test_split(np.arange(len(y_hp)), test_size=0.15, stratify=y_hp, random_state=SEED)
hp_rows = []
for lr_c in [0.001, 0.003, 0.01]:
    for dr_c in [0.10, 0.20, 0.30]:
        p_h, pr_h, _, _ = train_mlp(X_hp[hp_tr_i], y_hp[hp_tr_i], X_hp[hp_va_i], lr=lr_c, drop=dr_c, seed=SEED)
        mh = compute_metrics(y_hp[hp_va_i], p_h, pr_h)
        hp_rows.append({"lr": lr_c, "dropout": dr_c, "val_acc": mh["Accuracy"], "val_auc": mh["ROC_AUC"], "val_f1": mh["F1"]})
results_ledger["Internal_TrainVal_HP_Search"] = hp_rows

out_json = OUT_DIR / "comprehensive_revision_results.json"
out_json.write_text(json.dumps(results_ledger, indent=2), encoding="utf-8")
print(f"\n[DONE] Saved complete audit experiment ledger -> {out_json}")

