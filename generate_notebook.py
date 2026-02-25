#!/usr/bin/env python3
"""
Generate RESEARCH-QUALITY HSI Diffusion notebook.
Fixes applied (per expert review):
  1. No hardcoded results — all tables/charts from computed CSVs
  2. Single conditional GAN (one model, class-conditioned)
  3. Single conditional Diffusion (one model, all minority classes)
  4. Train/Val/Test split (70/15/15)
  5. Use q_sample consistently in training loop
  6. Normalization documented & verified
  7. 1D-CNN evaluation added
  8. Dynamic plots (len(MINORITY) not hardcoded 3)
"""
import json

def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – SETUP
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
# 🌈 Conditional Diffusion Model for HSI Classification
## Indian Pines — 1:100 Class Imbalance | Research-Quality Implementation
**Runtime:** ~1 h on T4 GPU | Splits: 70/15/15 | Single conditional model
"""))

cells.append(code("""\
# ── Cell 1: Setup ─────────────────────────────────────────────
import subprocess, sys
for p in ["scipy","scikit-learn","imbalanced-learn","tqdm","seaborn","pandas","joblib"]:
    subprocess.run([sys.executable,"-m","pip","install","-q",p])

import os, random, math, warnings
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.cuda.amp as amp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib, zipfile, urllib.request, scipy.io

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type=="cuda": print(torch.cuda.get_device_name(0))

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = "/workspace"
DATA_DIR    = f"{BASE_DIR}/data"
MODEL_DIR   = f"{BASE_DIR}/models"
FIG_DIR     = f"{BASE_DIR}/results/figures"
TABLE_DIR   = f"{BASE_DIR}/results/tables"
RESULT_DIR  = f"{BASE_DIR}/results"
for d in [DATA_DIR, MODEL_DIR, FIG_DIR, TABLE_DIR]: os.makedirs(d, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────
T           = 200    # diffusion timesteps
GAN_EPOCHS  = 50     # cGAN epochs
DIFF_EPOCHS = 100    # diffusion epochs
GEN_SAMPLES = 500    # synthetic samples per minority class
MINORITY    = [1, 7, 9]  # Alfalfa, Grass-mowed, Oats
Z_DIM       = 64     # latent dim (was 20, reviewer said too small)
print(f"✅ Config: T={T}, GAN_EPOCHS={GAN_EPOCHS}, DIFF_EPOCHS={DIFF_EPOCHS}, Z_DIM={Z_DIM}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – DATA
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Section 2 – Dataset Preparation"))

cells.append(code("""\
# ── Cell 2: Download Indian Pines ─────────────────────────────
DATA_URL  = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
LABEL_URL = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"

def download(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}…")
        urllib.request.urlretrieve(url, path)
    else: print(f"Cached: {path}")

download(DATA_URL,  f"{DATA_DIR}/ip_corrected.mat")
download(LABEL_URL, f"{DATA_DIR}/ip_gt.mat")

HSI   = scipy.io.loadmat(f"{DATA_DIR}/ip_corrected.mat")["indian_pines_corrected"].astype(np.float32)
LABEL = scipy.io.loadmat(f"{DATA_DIR}/ip_gt.mat")["indian_pines_gt"].astype(np.int32)

CLASS_NAMES = {
    0:"Background",1:"Alfalfa",2:"Corn-notill",3:"Corn-mintill",4:"Corn",
    5:"Grass-pasture",6:"Grass-trees",7:"Grass-mowed",8:"Hay-windrowed",
    9:"Oats",10:"Soybean-notill",11:"Soybean-mintill",12:"Soybean-clean",
    13:"Wheat",14:"Woods",15:"Bldgs-Grass",16:"Stone-Steel"
}
print(f"HSI: {HSI.shape}  |  Labels: {LABEL.shape}  |  Classes: {LABEL.max()}")
"""))

cells.append(code("""\
# ── Cell 3: 1:100 Imbalance ───────────────────────────────────
X_all = HSI.reshape(-1, HSI.shape[2]); y_all = LABEL.reshape(-1)
mask  = y_all > 0; X_all, y_all = X_all[mask], y_all[mask]

MINORITY_CAP = 100; MAJORITY_CAP = 5000
Xs, ys = [], []
for c in range(1,17):
    idx = np.where(y_all==c)[0]
    cap = MINORITY_CAP if c in MINORITY else min(MAJORITY_CAP, len(idx))
    ch  = np.random.choice(idx, min(cap,len(idx)), replace=False)
    Xs.append(X_all[ch]); ys.append(y_all[ch])

X_imb = np.vstack(Xs); y_imb = np.concatenate(ys)
for c in range(1,17):
    print(f"  Class {c:2d} {CLASS_NAMES[c]:<22}: {(y_imb==c).sum():5d}{'  ← MINORITY' if c in MINORITY else ''}")

fig, ax = plt.subplots(figsize=(13,4))
cols = ["#e74c3c" if c in MINORITY else "#3498db" for c in range(1,17)]
ax.bar([CLASS_NAMES[c][:10] for c in range(1,17)],
       [(y_imb==c).sum() for c in range(1,17)], color=cols)
ax.set_title("Class Distribution — 1:100 Imbalance (red=minority)")
ax.set_ylabel("Samples"); plt.xticks(rotation=45,ha="right"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_class_distribution.png",dpi=150); plt.show()
"""))

cells.append(code("""\
# ── Cell 4: PCA → Normalize → Train/Val/Test Split ───────────
# FIX: train/val/test (70/15/15) — reviewer correctly noted no val set

pca    = PCA(n_components=30, random_state=SEED)
scaler = StandardScaler()                  # ensures mean=0, std=1

X_pca  = pca.fit_transform(X_imb)
X_sc   = scaler.fit_transform(X_pca)

# Verify normalization (reviewer issue #7)
print(f"Mean  : {X_sc.mean():.6f}  (should be ~0)")
print(f"Std   : {X_sc.std():.6f}   (should be ~1)")
print(f"Min   : {X_sc.min():.3f}   Max: {X_sc.max():.3f}")

joblib.dump(pca,    f"{MODEL_DIR}/pca.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

# 70 / 15 / 15 stratified
X_tv, X_te, y_tv, y_te = train_test_split(X_sc, y_imb, test_size=0.15, stratify=y_imb, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv,  test_size=0.15/0.85, stratify=y_tv, random_state=SEED)

np.savez_compressed(f"{DATA_DIR}/preprocessed.npz",
    X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_te)
print(f"\\nExplained variance (30 PCs): {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
print("✅ Preprocessed data saved")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – BASELINES
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Section 3 – Baseline Experiments"))

cells.append(code("""\
# ── Cell 5: Metrics Helper + Baselines ───────────────────────
data = np.load(f"{DATA_DIR}/preprocessed.npz")
X_tr, X_va, X_te = data["X_tr"], data["X_va"], data["X_te"]
y_tr, y_va, y_te = data["y_tr"], data["y_va"], data["y_te"]

def get_metrics(y_true, y_pred, label=""):
    oa    = accuracy_score(y_true, y_pred)*100
    aa    = np.mean([accuracy_score(y_true[y_true==c], y_pred[y_true==c])*100
                     for c in np.unique(y_true)])
    kappa = cohen_kappa_score(y_true, y_pred)
    min_f1= np.mean([f1_score((y_true==c).astype(int),(y_pred==c).astype(int))
                     for c in MINORITY])
    print(f"{label:<30} OA={oa:.2f}%  AA={aa:.2f}%  κ={kappa:.3f}  MinF1={min_f1:.3f}")
    return {"Method":label,"OA%":round(oa,2),"AA%":round(aa,2),"Kappa":round(kappa,3),"MinF1":round(min_f1,3)}

results_all = []

print("Baseline SVM…")
svm_b = SVC(kernel="rbf",C=100,gamma="scale",random_state=SEED).fit(X_tr, y_tr)
results_all.append(get_metrics(y_te, svm_b.predict(X_te), "Baseline SVM"))

print("Baseline RF…")
rf_b  = RandomForestClassifier(100,max_depth=15,random_state=SEED,n_jobs=-1).fit(X_tr,y_tr)
results_all.append(get_metrics(y_te, rf_b.predict(X_te),  "Baseline RF"))

# Confusion matrix
fig,ax=plt.subplots(figsize=(8,6))
cm=confusion_matrix(y_te, svm_b.predict(X_te), labels=range(1,17))
sns.heatmap(cm,ax=ax,cmap="Reds",fmt="d",xticklabels=range(1,17),yticklabels=range(1,17))
ax.set_title("Baseline SVM – Catastrophic Minority Failure")
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_baseline_confusion.png",dpi=150); plt.show()
print("✅ Baseline done")
"""))

cells.append(code("""\
# ── Cell 6: SMOTE Baseline ────────────────────────────────────
try:
    X_sm, y_sm = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(X_tr, y_tr)
    svm_sm = SVC(kernel="rbf",C=100,gamma="scale",random_state=SEED).fit(X_sm, y_sm)
    results_all.append(get_metrics(y_te, svm_sm.predict(X_te), "SMOTE+SVM"))
    np.savez_compressed(f"{DATA_DIR}/smote_augmented.npz", X=X_sm, y=y_sm)
    print("✅ SMOTE done")
except Exception as e:
    print(f"SMOTE error: {e}")
    # Fallback: save train as-is so Cell 14 never crashes on missing file
    np.savez_compressed(f"{DATA_DIR}/smote_augmented.npz", X=X_tr, y=y_tr)
    results_all.append({"Method":"SMOTE+SVM","OA%":None,"AA%":None,"Kappa":None,"MinF1":None})
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – CONDITIONAL GAN (single model)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Section 4 – Conditional GAN (single model, class-conditioned)\n> **Fix #2:** One cGAN shared across all minority classes, not one per class."))

cells.append(code("""\
# ── Cell 7: Conditional GAN Architecture ─────────────────────
# FIX: single model conditioned on class label (reviewer issue #2)
# FIX: Z_DIM=64 (reviewer issue #8 — 20 was too small)

class ConditionalGenerator(nn.Module):
    def __init__(self, z=64, n_cls=17, emb=32, out=30):
        super().__init__()
        self.ce = nn.Embedding(n_cls, emb)
        self.net = nn.Sequential(
            nn.Linear(z+emb, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, out), nn.Tanh())
    def forward(self, z, c): return self.net(torch.cat([z, self.ce(c)], -1))

class ConditionalDiscriminator(nn.Module):
    def __init__(self, n_cls=17, emb=32, inp=30):
        super().__init__()
        self.ce = nn.Embedding(n_cls, emb)
        self.net = nn.Sequential(
            nn.Linear(inp+emb, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x, c): return self.net(torch.cat([x, self.ce(c)], -1))

G = ConditionalGenerator(Z_DIM); D = ConditionalDiscriminator()
print(f"cGAN Generator    params: {sum(p.numel() for p in G.parameters()):,}")
print(f"cGAN Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
"""))

cells.append(code("""\
# ── Cell 8: Conditional GAN Training (~8 min) ─────────────────
data = np.load(f"{DATA_DIR}/preprocessed.npz")
X_tr_np, y_tr_np = data["X_tr"], data["y_tr"]

# Build dataset of ALL minority training samples
mask_min  = np.isin(y_tr_np, MINORITY)
X_min_all = torch.tensor(X_tr_np[mask_min], dtype=torch.float32)
y_min_all = torch.tensor(y_tr_np[mask_min], dtype=torch.long)

dl  = DataLoader(TensorDataset(X_min_all, y_min_all), batch_size=32, shuffle=True)
G   = ConditionalGenerator(Z_DIM).to(DEVICE)
D   = ConditionalDiscriminator().to(DEVICE)
og  = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
od  = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))
bce = nn.BCELoss(); log_d, log_g = [], []

print(f"cGAN: {len(X_min_all)} samples across {len(MINORITY)} minority classes")
for ep in tqdm(range(GAN_EPOCHS), desc="cGAN"):
    dL=gL=0
    for xb, cb in dl:
        xb=xb.to(DEVICE); cb=cb.to(DEVICE); bs=xb.size(0)
        rl=torch.ones(bs,1,device=DEVICE); fl=torch.zeros(bs,1,device=DEVICE)
        z=torch.randn(bs,Z_DIM,device=DEVICE)
        ld=bce(D(xb,cb),rl)+bce(D(G(z,cb).detach(),cb),fl)
        od.zero_grad(); ld.backward(); od.step()
        z=torch.randn(bs,Z_DIM,device=DEVICE)
        lg=bce(D(G(z,cb),cb),rl)
        og.zero_grad(); lg.backward(); og.step()
        dL+=ld.item(); gL+=lg.item()
    log_d.append(dL/len(dl)); log_g.append(gL/len(dl))

torch.save(G.state_dict(), f"{MODEL_DIR}/cgan_generator.pth")
print(f"✅ cGAN trained | final D={log_d[-1]:.3f}  G={log_g[-1]:.3f}")

fig,ax=plt.subplots(figsize=(7,3))
ax.plot(log_d,label="D"); ax.plot(log_g,label="G")
ax.set_title("Conditional GAN Training Loss"); ax.legend(); ax.set_xlabel("Epoch")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig3_cgan_training.png",dpi=150)
plt.show()
"""))

cells.append(code("""\
# ── Cell 9: cGAN Sample Generation & Evaluation ──────────────
data = np.load(f"{DATA_DIR}/preprocessed.npz")
X_tr_np, X_te, y_tr_np, y_te = data["X_tr"], data["X_te"], data["y_tr"], data["y_te"]

G = ConditionalGenerator(Z_DIM).to(DEVICE)
G.load_state_dict(torch.load(f"{MODEL_DIR}/cgan_generator.pth", map_location=DEVICE))
G.eval()

Xs_gan, ys_gan = [], []
with torch.no_grad():
    for cls in MINORITY:
        c_t = torch.full((GEN_SAMPLES,), cls, dtype=torch.long, device=DEVICE)
        z   = torch.randn(GEN_SAMPLES, Z_DIM, device=DEVICE)
        syn = G(z, c_t).cpu().numpy()
        Xs_gan.append(syn); ys_gan.append(np.full(GEN_SAMPLES, cls))
        print(f"  GAN cls {cls}: {syn.shape}")

np.savez_compressed(f"{DATA_DIR}/gan_samples.npz",
                    X=np.vstack(Xs_gan), y=np.concatenate(ys_gan))

X_gan_aug = np.vstack([X_tr_np]+Xs_gan)
y_gan_aug = np.concatenate([y_tr_np]+ys_gan)
svm_g = SVC(kernel="rbf",C=100,gamma="scale",random_state=SEED).fit(X_gan_aug, y_gan_aug)
results_all.append(get_metrics(y_te, svm_g.predict(X_te), "cGAN+SVM"))
print("✅ cGAN eval done")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – CONDITIONAL DIFFUSION (single model)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Section 5 – Conditional Diffusion Model (single, shared)\n> **Fix #3:** One diffusion model trained on all minority classes together."))

cells.append(code("""\
# ── Cell 10: Noise Schedule ─────────────────────────────────
beta      = torch.linspace(1e-4, 0.02, T)
alpha     = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
sqrt_ab   = alpha_bar.sqrt()
sqrt_1mab = (1.0 - alpha_bar).sqrt()

# FIX #5: single canonical q_sample — never duplicate this formula
def q_sample(x0, t):
    \"\"\"Forward diffusion q(x_t | x_0) = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε\"\"\"
    eps = torch.randn_like(x0)
    sab = sqrt_ab[t.cpu()].view(-1,1).to(x0.device)
    s1m = sqrt_1mab[t.cpu()].view(-1,1).to(x0.device)
    return sab*x0 + s1m*eps, eps   # returns (x_t, noise)

# Verify forward diffusion
data = np.load(f"{DATA_DIR}/preprocessed.npz")
X_s  = torch.tensor(data["X_tr"][:1], dtype=torch.float32)
fig,axes=plt.subplots(1,5,figsize=(15,3))
for ax,tv in zip(axes,[0,50,100,150,199]):
    xt,_=q_sample(X_s, torch.tensor([tv]))
    ax.plot(xt[0].numpy()); ax.set_title(f"t={tv}"); ax.set_xlabel("PC Dim")
fig.suptitle("Forward Diffusion — Spectral Degradation"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_forward_diffusion.png",dpi=150); plt.show()
print("✅ Noise schedule ready")
"""))

cells.append(code("""\
# ── Cell 11: Conditional Denoiser ───────────────────────────
class SinEmbed(nn.Module):
    def __init__(self, dim=128):
        super().__init__(); self.dim=dim
    def forward(self, t):
        half=self.dim//2
        freq=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half)
        arg =t.float().unsqueeze(1)*freq.unsqueeze(0)
        return torch.cat([arg.sin(), arg.cos()], dim=-1)

class ConditionalDenoiser(nn.Module):
    \"\"\"MLP-based conditional denoiser (spectral domain).\"\"\"
    def __init__(self, x_dim=30, t_dim=128, c_dim=128, n_cls=17):
        super().__init__()
        self.te = SinEmbed(t_dim); self.ce = nn.Embedding(n_cls, c_dim)
        in_d = x_dim+t_dim+c_dim
        self.net = nn.Sequential(
            nn.Linear(in_d,512), nn.GroupNorm(8,512), nn.SiLU(),
            nn.Linear(512,256),  nn.GroupNorm(8,256), nn.SiLU(),
            nn.Linear(256,128),  nn.GroupNorm(8,128), nn.SiLU(),
            nn.Linear(128,256),  nn.GroupNorm(8,256), nn.SiLU(),
            nn.Linear(256,512),  nn.GroupNorm(8,512), nn.SiLU(),
            nn.Linear(512, x_dim))
    def forward(self, x, t, c):
        return self.net(torch.cat([x, self.te(t), self.ce(c)], dim=-1))

m=ConditionalDenoiser()
print(f"Denoiser params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
out=m(torch.randn(4,30), torch.randint(0,T,(4,)), torch.randint(0,17,(4,)))
print(f"Output shape: {out.shape}"); del m
"""))

cells.append(code("""\
# ── Cell 12: Single Conditional Diffusion Training (~40 min) ──
# FIX #3: train ONE model on ALL minority classes simultaneously
# FIX #4: always call q_sample — never duplicate formula

data = np.load(f"{DATA_DIR}/preprocessed.npz")
X_tr_np, y_tr_np = data["X_tr"], data["y_tr"]

# All minority training samples in one dataset
mask_min  = np.isin(y_tr_np, MINORITY)
X_min_all = torch.tensor(X_tr_np[mask_min], dtype=torch.float32)
y_min_all = torch.tensor(y_tr_np[mask_min], dtype=torch.long)
print(f"Minority training samples: {len(X_min_all)}  across classes {MINORITY}")

dl  = DataLoader(TensorDataset(X_min_all, y_min_all), batch_size=64,
                 shuffle=True, pin_memory=True)
model     = ConditionalDenoiser().to(DEVICE)
opt       = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
scaler_a  = amp.GradScaler()
mse       = nn.MSELoss(); log_loss = []

for ep in tqdm(range(DIFF_EPOCHS), desc="Diffusion"):
    ep_loss = 0
    for x0, c in dl:
        x0=x0.to(DEVICE); c=c.to(DEVICE)
        t = torch.randint(0, T, (x0.size(0),), device=DEVICE)
        # FIX #4: canonical q_sample (not re-implemented inline)
        xt, eps = q_sample(x0, t)
        with amp.autocast():
            loss = mse(eps, model(xt, t, c))
        scaler_a.scale(loss).backward()
        scaler_a.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_a.step(opt); scaler_a.update(); opt.zero_grad()
        ep_loss += loss.item()
    log_loss.append(ep_loss/len(dl))

torch.save(model.state_dict(), f"{MODEL_DIR}/diffusion_model.pth")
if DEVICE.type=="cuda": torch.cuda.empty_cache()

fig,ax=plt.subplots(figsize=(8,3))
ax.plot(log_loss); ax.set_title("Diffusion Training Loss (single conditional model)")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_diffusion_training.png",dpi=150); plt.show()
print(f"✅ Diffusion trained | final loss={log_loss[-1]:.5f}")
"""))

cells.append(code("""\
# ── Cell 13: Reverse Diffusion Sampling (~2 min) ──────────────
@torch.no_grad()
def sample(model, n, cls_label, x_dim=30):
    model.eval()
    c  = torch.full((n,), cls_label, dtype=torch.long, device=DEVICE)
    xt = torch.randn(n, x_dim, device=DEVICE)
    for tv in reversed(range(T)):
        t_v  = torch.full((n,), tv, dtype=torch.long, device=DEVICE)
        ep_p = model(xt, t_v, c)
        at=alpha[tv].to(DEVICE); abt=alpha_bar[tv].to(DEVICE); bt=beta[tv].to(DEVICE)
        mu = (1/at.sqrt())*(xt - (bt/(1-abt).sqrt())*ep_p)
        xt = mu + bt.sqrt()*torch.randn_like(xt) if tv>0 else mu
    return xt.cpu().numpy()

# Load single shared model
diff_model = ConditionalDenoiser().to(DEVICE)
diff_model.load_state_dict(torch.load(f"{MODEL_DIR}/diffusion_model.pth", map_location=DEVICE))

Xs_diff, ys_diff = [], []
for cls in MINORITY:
    syn = sample(diff_model, GEN_SAMPLES, cls)
    Xs_diff.append(syn); ys_diff.append(np.full(GEN_SAMPLES, cls))
    print(f"  Class {cls}: generated {syn.shape}")
if DEVICE.type=="cuda": torch.cuda.empty_cache()

Xs_diff = np.vstack(Xs_diff); ys_diff = np.concatenate(ys_diff)
np.savez_compressed(f"{DATA_DIR}/diffusion_samples.npz", X=Xs_diff, y=ys_diff)
print(f"✅ Saved {len(ys_diff)} synthetic samples")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Section 6 – Evaluation: SVM, RF, and 1D-CNN\n> **Fix #6:** 1D-CNN added as standard HSI classifier."))

cells.append(code("""\
# ── Cell 14: SVM / RF Evaluation ─────────────────────────────
data      = np.load(f"{DATA_DIR}/preprocessed.npz")
X_tr_np, X_te, y_tr_np, y_te = data["X_tr"], data["X_te"], data["y_tr"], data["y_te"]

diff_syn  = np.load(f"{DATA_DIR}/diffusion_samples.npz")
gan_syn   = np.load(f"{DATA_DIR}/gan_samples.npz")
smote_aug = np.load(f"{DATA_DIR}/smote_augmented.npz")

X_diff_aug = np.vstack([X_tr_np, diff_syn["X"]]); y_diff_aug = np.concatenate([y_tr_np, diff_syn["y"]])
X_gan_aug  = np.vstack([X_tr_np, gan_syn["X"]]);  y_gan_aug  = np.concatenate([y_tr_np, gan_syn["y"]])

for name, clf, Xfit, yfit in [
    ("Diffusion+SVM", SVC(kernel="rbf",C=100,gamma="scale",random_state=SEED), X_diff_aug, y_diff_aug),
    ("Diffusion+RF",  RandomForestClassifier(100,max_depth=15,random_state=SEED,n_jobs=-1), X_diff_aug, y_diff_aug)]:
    clf.fit(Xfit, yfit)
    results_all.append(get_metrics(y_te, clf.predict(X_te), name))

# Confusion matrix (diffusion+SVM)
svm_d = SVC(kernel="rbf",C=100,gamma="scale",random_state=SEED).fit(X_diff_aug, y_diff_aug)
fig,ax=plt.subplots(figsize=(8,6))
cm=confusion_matrix(y_te, svm_d.predict(X_te), labels=range(1,17))
sns.heatmap(cm,ax=ax,cmap="Greens",fmt="d",xticklabels=range(1,17),yticklabels=range(1,17))
ax.set_title("Diffusion+SVM Confusion Matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_diffusion_confusion.png",dpi=150); plt.show()

# Save cumulative results so far
pd.DataFrame([r for r in results_all if r["OA%"] is not None]).to_csv(
    f"{RESULT_DIR}/all_results.csv", index=False)
print("✅ SVM/RF evaluation done")
"""))

cells.append(code("""\
# ── Cell 15: 1D-CNN Evaluation (Fix #6) ──────────────────────
# FIX #6: add standard 1D CNN classifier (required for HSI paper)

class CNN1D(nn.Module):
    def __init__(self, seq_len=30, n_cls=17):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        with torch.no_grad():
            dummy = torch.zeros(1,1,seq_len); out = self.conv(dummy)
            flat  = out.flatten(1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_cls))
    def forward(self, x): return self.fc(self.conv(x.unsqueeze(1)).flatten(1))

def train_cnn(X, y, X_te, y_te, label, epochs=50):
    classes  = np.unique(y)
    # remap labels to 0-indexed
    lmap = {c:i for i,c in enumerate(sorted(np.unique(np.concatenate([y,y_te]))))}
    y_r  = np.array([lmap[v] for v in y])
    yt_r = np.array([lmap[v] for v in y_te])
    n_cls = len(lmap)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(y_r, dtype=torch.long)
    dl = DataLoader(TensorDataset(Xt,Yt), batch_size=128, shuffle=True)
    net = CNN1D(X.shape[1], n_cls).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        for xb,yb in dl:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            loss=F.cross_entropy(net(xb),yb)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(X_te,dtype=torch.float32,device=DEVICE))
        preds_idx = logits.argmax(-1).cpu().numpy()
    inv = {i:c for c,i in lmap.items()}
    preds = np.array([inv[i] for i in preds_idx])
    return get_metrics(y_te, preds, label)

data = np.load(f"{DATA_DIR}/preprocessed.npz")
X_tr_np, X_te, y_tr_np, y_te = data["X_tr"], data["X_te"], data["y_tr"], data["y_te"]
diff_syn = np.load(f"{DATA_DIR}/diffusion_samples.npz")
X_diff_aug = np.vstack([X_tr_np, diff_syn["X"]]); y_diff_aug = np.concatenate([y_tr_np, diff_syn["y"]])

print("Training 1D-CNN on baseline data…")
r_cnn_base = train_cnn(X_tr_np, y_tr_np, X_te, y_te, "Baseline 1D-CNN", epochs=50)
print("Training 1D-CNN on Diffusion-augmented data…")
r_cnn_diff = train_cnn(X_diff_aug, y_diff_aug, X_te, y_te, "Diffusion+1D-CNN", epochs=50)
results_all += [r_cnn_base, r_cnn_diff]

# Re-save with CNN results
df_all = pd.DataFrame([r for r in results_all if r["OA%"] is not None])
df_all.to_csv(f"{RESULT_DIR}/all_results.csv", index=False)
print("✅ 1D-CNN evaluation done"); print(df_all.to_string())
"""))

cells.append(code("""\
# ── Cell 16: Spectral Fidelity (SAM + KS) ────────────────────
from scipy.stats import ks_2samp

def sam_score(r, s):
    d = np.sum(r*s,axis=1)/(np.linalg.norm(r,axis=1)*np.linalg.norm(s,axis=1)+1e-9)
    return np.arccos(np.clip(d,-1,1)).mean()

data    = np.load(f"{DATA_DIR}/preprocessed.npz")
syn_d   = np.load(f"{DATA_DIR}/diffusion_samples.npz")
syn_g   = np.load(f"{DATA_DIR}/gan_samples.npz")
X_tr_np, y_tr_np = data["X_tr"], data["y_tr"]

fid_rows = []
for cls in MINORITY:
    real    = X_tr_np[y_tr_np==cls]
    diff_s  = syn_d["X"][syn_d["y"]==cls]
    gan_s   = syn_g["X"][syn_g["y"]==cls]
    n       = min(len(real), len(diff_s), len(gan_s))
    sam_d   = sam_score(real[:n], diff_s[:n])
    sam_g   = sam_score(real[:n], gan_s[:n])
    ks_d    = ks_2samp(real.flatten(), diff_s.flatten()).pvalue
    ks_g    = ks_2samp(real.flatten(), gan_s.flatten()).pvalue
    fid_rows.append({"Class":cls,"Diffusion_SAM":round(sam_d,4),"GAN_SAM":round(sam_g,4),
                     "Diffusion_KS_p":round(ks_d,4),"GAN_KS_p":round(ks_g,4)})
    print(f"Cls {cls}  Diff SAM={sam_d:.4f} KS_p={ks_d:.4f}  |  GAN SAM={sam_g:.4f} KS_p={ks_g:.4f}")

pd.DataFrame(fid_rows).to_csv(f"{RESULT_DIR}/spectral_fidelity.csv", index=False)

# Spectral comparison — FIX #14: use len(MINORITY), not hardcoded 3
n_min = len(MINORITY)
fig,axes=plt.subplots(2, n_min, figsize=(5*n_min, 7))
if n_min==1: axes = [[axes[0]], [axes[1]]]
for col,cls in enumerate(MINORITY):
    real   = X_tr_np[y_tr_np==cls]
    diff_s = syn_d["X"][syn_d["y"]==cls]
    for i in range(min(8,len(real))): axes[0][col].plot(real[i],alpha=0.4,color="steelblue")
    for i in range(min(8,len(diff_s))): axes[1][col].plot(diff_s[i],alpha=0.4,color="coral",ls="--")
    axes[0][col].set_title(f"Real Cls {cls}"); axes[1][col].set_title(f"Diffusion Cls {cls}")
axes[0][0].set_ylabel("Amplitude"); axes[1][0].set_ylabel("Amplitude")
plt.suptitle("Spectral Fidelity: Real vs Diffusion"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_spectral_fidelity.png",dpi=150); plt.show()
print("✅ Spectral fidelity done")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – TABLES & PLOTS (from real computed results)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Section 7 – Publication Tables & Figures\n> **Fix #1:** All tables and charts use real computed metrics — no hardcoded values."))

cells.append(code("""\
# ── Cell 17: Generate Tables from REAL results ────────────────
# FIX #1: NEVER hardcode results — always read from CSV

os.makedirs(TABLE_DIR, exist_ok=True)
df_all = pd.read_csv(f"{RESULT_DIR}/all_results.csv")
fid_df = pd.read_csv(f"{RESULT_DIR}/spectral_fidelity.csv")

print("\\n📋 ACTUAL COMPUTED RESULTS:")
print(df_all.to_string(index=False))

# Save LaTeX tables from real data
df_all.to_csv(f"{TABLE_DIR}/table1_classification_results.csv", index=False)
with open(f"{TABLE_DIR}/table1_classification_results.tex","w") as f:
    f.write(df_all.to_latex(index=False,escape=False,float_format="%.2f"))

fid_df.to_csv(f"{TABLE_DIR}/table2_spectral_fidelity.csv", index=False)
with open(f"{TABLE_DIR}/table2_spectral_fidelity.tex","w") as f:
    f.write(fid_df.to_latex(index=False,escape=False,float_format="%.4f"))

print("\\n✅ Tables saved from real computed results (no hardcoded values)")
"""))

cells.append(code("""\
# ── Cell 18: Dynamic Comparison Bar Charts (from real CSV) ────
# FIX #1: bar charts built from computed df_all — not hardcoded lists

df_all = pd.read_csv(f"{RESULT_DIR}/all_results.csv")
methods = df_all["Method"].tolist()
oas     = df_all["OA%"].tolist()
min_f1  = df_all["MinF1"].tolist()

cmap = plt.cm.get_cmap("tab10", len(methods))
cols = [cmap(i) for i in range(len(methods))]

fig,axes=plt.subplots(1,2,figsize=(max(10,len(methods)*1.4),5))
axes[0].bar(methods, oas, color=cols)
for i,v in enumerate(oas): axes[0].text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
axes[0].set_title("Overall Accuracy (%)"); axes[0].set_ylabel("OA (%)"); axes[0].set_ylim(0,100)
plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")

axes[1].bar(methods, min_f1, color=cols)
for i,v in enumerate(min_f1): axes[1].text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
axes[1].set_title("Minority Class F1"); axes[1].set_ylabel("F1"); axes[1].set_ylim(0,1)
plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

plt.suptitle("All Methods — Indian Pines (1:100 Imbalance)"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig8_final_comparison.png",dpi=150); plt.show()
print("📊 fig8 saved — from REAL computed results")
"""))

cells.append(code("""\
# ── Cell 19: Package & Export ─────────────────────────────────
df_all = pd.read_csv(f"{RESULT_DIR}/all_results.csv")
best   = df_all.loc[df_all["OA%"].idxmax()]

summary = f\"\"\"
# HSI Diffusion Model — Results Summary

## Actual Computed Results
Best method: {best['Method']}
Best OA    : {best['OA%']:.2f}%
Best MinF1 : {best['MinF1']:.3f}

## All Methods
{df_all.to_string(index=False)}

## Spectral Fidelity
{pd.read_csv(f'{RESULT_DIR}/spectral_fidelity.csv').to_string(index=False)}

## Methodology Notes
- Single conditional GAN (class-conditioned, Z_DIM=64)
- Single conditional Diffusion (shared across all minority classes)
- Train/Val/Test split: 70/15/15 (stratified)
- Evaluators: SVM, RF, 1D-CNN
- Fidelity: SAM + KS-test per minority class
\"\"\"
with open(f"{RESULT_DIR}/RESULTS_SUMMARY.md","w") as f: f.write(summary)

with zipfile.ZipFile(f"{BASE_DIR}/diffusion_hsi_results.zip","w",zipfile.ZIP_DEFLATED) as zf:
    for root,_,files in os.walk(f"{RESULT_DIR}"):
        for fn in files:
            fp=os.path.join(root,fn); zf.write(fp,os.path.relpath(fp,BASE_DIR))
    for fn in os.listdir(MODEL_DIR):
        fp=os.path.join(MODEL_DIR,fn); zf.write(fp,os.path.join("models",fn))

sz=os.path.getsize(f"{BASE_DIR}/diffusion_hsi_results.zip")/1e6
print(f"📦 Archive: {BASE_DIR}/diffusion_hsi_results.zip  ({sz:.1f} MB)")
print(f\"\"\"
╔══════════════════════════════════════════════════════════════╗
║  ✅  EXPERIMENT COMPLETE — Research Quality                 ║
║                                                              ║
║  Best: {best['Method']:<34} OA={best['OA%']:.2f}%  ║
║  Results: REAL computed (no hardcoded values)               ║
║  Models : 1 cGAN + 1 Diffusion (conditional)               ║
║  Splits : 70/15/15 train/val/test                           ║
╚══════════════════════════════════════════════════════════════╝
\"\"\")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name":"Python 3","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
        "accelerator": "GPU"
    },
    "cells": cells
}

OUT = "/Users/yash/.gemini/antigravity/scratch/hsi_diffusion/hsi_diffusion_model.ipynb"
with open(OUT,"w") as f: json.dump(nb, f, indent=1)
print(f"✅ Notebook written — {len(cells)} cells")
print(f"   Path: {OUT}")
