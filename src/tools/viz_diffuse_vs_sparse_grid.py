"""Overlay grid: baseline (DIFFUSE softmax) vs a215 (SPARSE entmax) attention on the same ROIs.
Shows what 'diffuse vs sparse' looks like on tissue -> illustrates why the diffuse read is preferred
(a215 is a failed sparse candidate, used here purely to visualise the contrast)."""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.simple_mil import SimpleGatedMIL
from models.novelty_attempts.a215_adaptive_sparse_gated_attention_pooling import Model as A215
BB="virchow2"; DIM=BACKBONE_CONFIG[BB]["dim"]; P=224; S=112; SQ=360
def load(m,pat):
    ck=sorted(glob.glob(str(ROOT/pat)))[-1]; sd=torch.load(ck,map_location="cpu",weights_only=False)
    if isinstance(sd,dict) and "model_state_dict" in sd: sd=sd["model_state_dict"]
    m.load_state_dict(sd); return m.eval()
def stitch(bag):
    rc=bag["rc"]; rc=rc.numpy() if hasattr(rc,"numpy") else np.array(rc)
    R,C=int(rc[:,0].max())+1,int(rc[:,1].max())+1; H,W=(R-1)*S+P,(C-1)*S+P
    cv=np.ones((H,W,3),np.float32)*0.95
    for (r,c),pp in zip(rc,bag["patch_paths"]):
        fp=pp if os.path.isabs(str(pp)) else str(ROOT/str(pp))
        if os.path.exists(fp): cv[r*S:r*S+P,c*S:c*S+P]=np.asarray(Image.open(fp).convert("RGB"),np.float32)/255.
    return cv,rc,R,C,H,W
def wmap(rc,w,H,W):
    R,C=int(rc[:,0].max())+1,int(rc[:,1].max())+1; g=np.zeros((R,C),np.float32)
    for (r,c),a in zip(rc,w): g[int(r),int(c)]=a
    return np.asarray(Image.fromarray(g).resize((W,H),Image.BILINEAR),np.float32)
def ov(cv,wm,gamma=0.7):
    n=wm/(wm.max()+1e-9); rgba=cm.get_cmap("inferno")(n)[...,:3].astype(np.float32)
    al=(n**gamma)[...,None]*0.85; gray=cv.mean(2,keepdims=True).repeat(3,2)*0.55+0.2
    return (1-al)*gray+al*rgba
def sq(a): return np.asarray(Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)).resize((SQ,SQ),Image.BILINEAR),np.float32)/255.
ds=GradingBagDatasetFull(ROOT/"data"/BACKBONE_CONFIG[BB]["feature_dir"]); te=patient_split(ds,seed=2)[2]
base=load(SimpleGatedMIL(input_dim=DIM,num_classes=1,topk=0),"experiments/20260603_ablation/patch/*simple_virchow2_regression*/best_*.pth")
spar=load(A215(input_dim=DIM,num_classes=1),"experiments/a215_ablation/regression_seed2/casc_a215_virchow2_s2_*/best_*.pth")
allg={g:[] for g in range(4)}
for i in te: allg[int(ds.samples[i][1])].append(i)
fig,ax=plt.subplots(4,3,figsize=(11,14.5))
for c,t in enumerate(["ROI (original)","baseline — DIFFUSE (softmax)","ASGAP — SPARSE (1.5-entmax)"]): ax[0,c].set_title(t,fontsize=12)
for g in range(4):
    i=allg[g][len(allg[g])//2]; pt,lab=ds.samples[i]
    bag=torch.load(pt,map_location="cpu",weights_only=False); feats=bag["feats"].float()
    with torch.no_grad():
        _,wb,_=base(feats,return_attention=True); _,ws,_=spar(feats,return_attention=True)
    wb=wb.flatten().numpy(); ws=ws.flatten().numpy()
    cv,rc,R,C,H,W=stitch(bag)
    nz=int((ws>1e-6).sum())
    cells=[sq(cv),sq(ov(cv,wmap(rc,wb,H,W))),sq(ov(cv,wmap(rc,ws,H,W)))]
    for c in range(3): ax[g,c].imshow(cells[c]); ax[g,c].set_xticks([]); ax[g,c].set_yticks([])
    ax[g,0].set_ylabel(f"G{lab}\nN={len(wb)}",fontsize=12,rotation=0,labelpad=30,va="center")
    ax[g,2].set_xlabel(f"{nz}/{len(ws)} patches kept",fontsize=9)
    print(f"G{lab}: N={len(wb)} sparse_kept={nz}/{len(ws)} base_maxw={wb.max():.3f} sparse_maxw={ws.max():.3f}")
fig.suptitle("DIFFUSE (baseline, used) vs SPARSE (ASGAP 1.5-entmax, failed) attention on tissue\n"
             "baseline spreads weight over the whole meshwork; entmax zeroes most patches (concentrated)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
out=ROOT/"results"/"diffuse_vs_sparse_grid.png"; fig.savefig(out,dpi=135); print(f"\nsaved -> {out}")
