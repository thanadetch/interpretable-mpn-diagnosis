"""Ad-hoc temperature sweep (NO retraining): re-temper a TRAINED baseline model's attention at
inference and measure test QWK vs T.

T only affects the final pooling: attn = softmax(e / T). So we load the trained ABMIL
baseline, compute the (frozen) attention energies e on the test bags ONCE, then re-pool with many T
and re-run the (frozen) classifier. Seconds, CPU, no GPU contention.

CAVEAT (honest): the attention scorer + classifier were TRAINED for T=1, so this measures the test
SENSITIVITY to re-tempering a fixed model — a fast directional probe, NOT equivalent to retraining
with that T (that is the separate a163 sweep). Still test-peeking => characterisation, not selectable.
Sanity: T=1.0 must reproduce the known baseline test_qwk (titan .9584 / virchow2 .9476 / uni2 .9418).
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.simple_mil import SimpleGatedMIL

BASE_GLOB = {
    "titan":    "experiments/20260603_ablation/patch/*simple_titan_regression*",
    "virchow2": "experiments/20260603_ablation/patch/*simple_virchow2_regression*",
    "uni2":     "experiments/20260603_ablation/patch/*simple_uni2_regression*",
}
KNOWN_BASE = {"titan": 0.9584, "virchow2": 0.9476, "uni2": 0.9418}
import os
if os.environ.get("A_TS"):
    TS = [float(x) for x in os.environ["A_TS"].split(",")]
else:
    TS = [0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def load_ckpt(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    return model


def sweep_backbone(bb):
    dim = BACKBONE_CONFIG[bb]["dim"]; fdir = BACKBONE_CONFIG[bb]["feature_dir"]
    ds = GradingBagDatasetFull(ROOT / "data" / fdir)
    test_idx = patient_split(ds, seed=2)[2]
    rundir = sorted(glob.glob(str(ROOT / BASE_GLOB[bb])))[-1]
    ckpt = sorted(glob.glob(rundir + "/best_*.pth"))[0]
    model = load_ckpt(SimpleGatedMIL(input_dim=dim, num_classes=1, topk=0), ckpt).eval()

    # precompute frozen energies e + bottleneck h per test bag (ONCE)
    bags = []
    labels = []
    with torch.no_grad():
        for i in test_idx:
            feat, label, _slide = ds[i]
            feat = feat.float()
            h = model.bottleneck(feat)
            e = model.attention_W(model.attention_V(h) * model.attention_U(h)).squeeze(-1)
            bags.append((h, e)); labels.append(int(label))

    print(f"\n### {bb.upper()}  (n_test={len(labels)}, dim={dim})  baseline≈{KNOWN_BASE[bb]:.4f}")
    print(f"  {'T':>6} {'test_qwk':>9} {'Δvs base':>9}  {'mean_attn_entropy':>17}")
    best = (None, -1)
    with torch.no_grad():
        for T in TS:
            preds = []; ents = []
            for h, e in bags:
                a = F.softmax(e / T, dim=0)
                z = torch.mv(h.t(), a)
                raw = model.classifier(z).view(-1).item()
                preds.append(int(max(0, min(3, round(raw)))))
                n = a.shape[0]
                if n > 1:
                    ent = -(a * a.clamp(min=1e-12).log()).sum().item() / torch.log(torch.tensor(float(n))).item()
                    ents.append(ent)
            qwk = cohen_kappa_score(labels, preds, labels=[0, 1, 2, 3], weights="quadratic")
            me = sum(ents) / len(ents) if ents else float("nan")
            mark = " <= T=1 (sanity)" if abs(T - 1.0) < 1e-9 else (" *BEST*" if qwk > best[1] else "")
            if qwk > best[1]: best = (T, qwk)
            print(f"  {T:>6.2f} {qwk:>9.4f} {qwk-KNOWN_BASE[bb]:>+9.4f}  {me:>17.3f}{mark}")
    print(f"  -> best ad-hoc T={best[0]} (qwk={best[1]:.4f}, Δ={best[1]-KNOWN_BASE[bb]:+.4f})")
    return best


def main():
    bbs = sys.argv[1:] or ["titan", "virchow2", "uni2"]
    for bb in bbs:
        sweep_backbone(bb)


if __name__ == "__main__":
    main()
