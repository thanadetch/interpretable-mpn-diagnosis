# Example Commands to Train, Evaluate, and Explain Models

## Training Commands
### Resize Data Mode
`python src/train.py --task classification --data_mode resize --model resnet18 --epochs 50`

`python src/train.py --task grading --data_mode resize --model resnet18 --epochs 50`

### Patch Data Mode
`python src/train.py --task classification --data_mode patch --model resnet18 --epochs 50`

`python src/train.py --task grading --data_mode patch --model resnet18 --epochs 50`

---

## Evaluation and Explanation Commands
### *Classification Task*
### Patch Data Mode
`python src/evaluate.py --checkpoint experiments/classification_resnet18_patch_20260105_105136/best_model.pth --data_mode patch`

`python src/explain.py --checkpoint experiments/classification_resnet18_patch_20260101_220104/best_model.pth --data_mode patch --num_samples 10`

### Resize Data Mode
`python src/evaluate.py --checkpoint experiments/classification_resnet18_resize_20260101_210945/best_model.pth --data_mode resize`

`python src/explain.py --checkpoint experiments/classification_resnet18_resize_20260101_210945/best_model.pth --data_mode resize --num_samples 10`

### *Grading Task*
### Patch Data Mode
`python src/evaluate.py --checkpoint experiments/grading_resnet18_patch_20260106_020449/best_model.pth --data_mode patch`

`python src/explain.py --checkpoint experiments/grading_resnet18_patch_20260101_173308/best_model.pth --data_mode patch --num_samples 10`

### Resize Data Mode
`python src/evaluate.py --checkpoint experiments/grading_resnet18_resize_20260101_164102/best_model.pth --data_mode resize`

`python src/explain.py --checkpoint experiments/grading_resnet18_resize_20260101_164102/best_model.pth --data_mode resize --num_samples 10`

