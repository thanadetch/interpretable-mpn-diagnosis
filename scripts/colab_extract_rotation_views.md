# Colab — rotation-view extraction (Virchow2, C4)

Run these six cells in order. Only cell 5 does real work; everything else is setup.
Decision record for *why* this is rotation-only and Virchow2-only:
`THESIS_AUGMENTATION_PLACEMENT.md` §5.

---

### 1. Repo + dependencies

```python
%cd /content
!git clone <YOUR-REPO-URL> interpretable-mpn-diagnosis
%cd /content/interpretable-mpn-diagnosis
!pip install -q timm transformers huggingface_hub tqdm
```

### 2. Hugging Face token (Virchow2 is gated)

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

### 3. Mount Drive and point `data/processed_grading` at the patches

```python
from google.colab import drive
drive.mount('/content/drive')
!ln -sfn "/content/drive/MyDrive/<YOUR-PATH>/processed_grading" data/processed_grading
```

### 4. Verify the plan before spending GPU time

```python
!python -m src.data.extract_reti_views --backbone virchow2 --views 1-3 --dry_run
```

Expect exactly: `Patients: 50 · Images: 1330 · Patches: 58550 · Est. size: ~0.45 GB`.
Anything else means the symlink in cell 3 is wrong — fix it before continuing.

### 5. Extract — the only commands that do real work

One command per backbone (`--views` loops inside the script; `--backbone` cannot, since it
is a different model). views 1-3 = rot90 / rot180 / rot270, the full 90-degree rotation group.

**Session A (~2.5 h) — Virchow2 then TITAN**

```python
!python -m src.data.extract_reti_views --backbone virchow2 --views 1-3 --device cuda --batch_size 128
!python -m src.data.extract_reti_views --backbone titan    --views 1-3 --device cuda --batch_size 128
```

**Session B (~3 h) — UNI2-h**

```python
!python -m src.data.extract_reti_views --backbone uni2 --views 1-3 --device cuda --batch_size 128
```

Order is deliberate. Virchow2 is first because it carries the hypothesis (lowest measured
rotation invariance, best-performing backbone here) — if it does not move, the other two can
be skipped entirely. UNI2-h is last and alone because it is roughly twice the cost of
Virchow2 per view and would otherwise be the thing a session cut-off destroys.

Every command is safe to re-run verbatim after a disconnect: finished `.pt` files are
skipped, so it resumes rather than restarting. If a free session keeps dropping inside
UNI2-h, split it further with `--views 1-2` then `--views 3`.

| backbone | per view (T4) | x3 views | size (fp16) |
|---|---|---|---|
| Virchow2 (ViT-H/14) | ~25-35 min | ~1.5 h | ~450 MB |
| TITAN (CONCHv1.5) | ~15-25 min | ~1 h | ~270 MB |
| UNI2-h (ViT-g/14) | ~50-70 min | ~3 h | ~540 MB |
| **total** | | **~5.5 h** | **~1.26 GB** |

### 6. Archive once, then copy to Drive

Run at the end of each session, for whichever backbones that session finished.

```python
!tar -czf /content/reti_rot_views.tar.gz -C data \
    features_virchow2_reti_views features_titan_reti_views features_uni2_reti_views
!cp /content/reti_rot_views.tar.gz /content/drive/MyDrive/
```

Do **not** point `--output_root` at Drive directly: that writes 3,990 small files over the
Drive FUSE mount and is several times slower than one archive.

---

### Back on the Mac

```bash
tar -xzf ~/Downloads/reti_rot_views.tar.gz -C data/
ls data/features_*_reti_views/            # expect view1 view2 view3 under each
```

Then train the full 6-cell grid in parallel (~5 min) and compare each cell against its own
no-augmentation run on both gates — Virchow2/ASGAP no-aug = 0.7976 val / 0.9588 test:

```bash
for bb in virchow2 uni2 titan; do
  python -m src.train_grading_reti --backbone $bb \
    --model_type novelty_attempt --novelty_id a215_learnable_entmax \
    --formulation regression --main_metric qwk --seed 2 \
    --device mps --num_workers 0 \
    --augmentation image_view --aug_strength 1.0 --prefix rot_as_$bb &
  python -m src.train_grading_reti --backbone $bb \
    --model_type simple \
    --formulation regression --main_metric qwk --seed 2 \
    --device mps --num_workers 0 \
    --augmentation image_view --aug_strength 1.0 --prefix rot_ab_$bb &
done; wait
```
