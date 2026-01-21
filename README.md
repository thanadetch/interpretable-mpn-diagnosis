# Example Commands to Train, Evaluate, and Explain Models

## Training Commands
### Resize Data Mode
`python src/train.py --task classification --data_mode resize --model resnet18 --epochs 50`

`python src/train.py --task grading --data_mode resize --model resnet18 --epochs 50`

### Patch Data Mode
`python src/train.py --task classification --data_mode patch --model resnet18 --epochs 50`
`python src/train.py --task classification --data_mode patch --model resnet18 --epochs 50`

`python src/train.py --task grading --data_mode patch --model resnet18 --epochs 50`
`python src/train.py --task grading --data_mode patch --model resnet50 --epochs 50`
---

## Evaluation and Explanation Commands
### *Classification Task*
### Patch Data Mode
`python src/evaluate.py --checkpoint experiments/classification_resnet18_patch_20260105_105136/best_model.pth --data_mode patch`
`python src/evaluate.py --checkpoint experiments/classification_resnet18_patch_20260112_151612/best_model.pth --data_mode patch`
`python src/evaluate.py --checkpoint experiments/classification_resnet18_patch_20260112_151612/best_model.pth --data_mode patch`
`python src/evaluate.py --checkpoint experiments/classification_densenet121_subtype_patch_clean_20260121_122850/best_model.pth --data_mode subtype_patch_clean  --aggregation mean --level image`

`python src/explain.py --checkpoint experiments/classification_resnet18_patch_20260101_220104/best_model.pth --data_mode patch --num_samples 10`

### Resize Data Mode
`python src/evaluate.py --checkpoint experiments/classification_resnet18_resize_20260101_210945/best_model.pth --data_mode resize`
`python src/evaluate.py --checkpoint experiments/classification_resnet18_resize_20260112_222224/best_model.pth --data_mode resize`

`python src/explain.py --checkpoint experiments/classification_resnet18_resize_20260101_210945/best_model.pth --data_mode resize --num_samples 10`
`python src/explain.py --checkpoint experiments/classification_resnet18_patch_20260112_151612/best_model.pth --data_mode patch --num_samples 30`
`python src/explain.py --checkpoint experiments/classification_densenet121_subtype_patch_clean_20260121_030818/best_model.pth --data_mode subtype_patch_clean --num_samples 50`

### *Grading Task*
### Patch Data Mode
`python src/evaluate.py --checkpoint experiments/grading_resnet18_patch_20260108_023815/best_model.pth --data_mode patch`
`python src/evaluate.py --checkpoint experiments/grading_densenet121_grading_patch_20260117_075925/best_model.pth --data_mode patch --aggregation clinical --level image`
`python src/evaluate.py --checkpoint experiments/grading_densenet121_grading_patch_clean_20260117_172917/best_model.pth --data_mode grading_patch_clean --aggregation clinical --level image`

`python src/evaluate.py --checkpoint experiments/grading_densenet121_grading_patch_clean_20260121_131139/best_model.pth --data_mode grading_patch_clean --aggregation clinical --level image`

`python src/explain.py --checkpoint experiments/grading_resnet18_patch_20260101_173308/best_model.pth --data_mode patch --num_samples 10`
`python src/explain.py --checkpoint experiments/grading_resnet18_patch_20260108_234850/best_model.pth --data_mode patch --num_samples 10`
`python src/explain.py --checkpoint experiments/grading_resnet18_patch_20260110_154314/best_model.pth --data_mode patch --num_samples 30`
`python src/explain.py --checkpoint experiments/grading_densenet121_grading_patch_20260117_075925/best_model.pth --data_mode patch --num_samples 50`
`python src/explain.py --checkpoint experiments/grading_densenet121_grading_patch_clean_20260117_172917/best_model.pth --data_mode grading_patch_clean --num_samples 50`
`python src/explain.py --checkpoint experiments/grading_densenet121_grading_patch_clean_20260121_131139/best_model.pth --data_mode grading_patch_clean --num_samples 50`

### Resize Data Mode
`python src/evaluate.py --checkpoint experiments/grading_resnet18_patch_20260108_101800/best_model.pth --data_mode resize`

`python src/explain.py --checkpoint experiments/grading_resnet18_resize_20260101_164102/best_model.pth --data_mode resize --num_samples 10`


### Data Preprocessing Command for Patch Mode
`python src/preprocess.py --patch_size 512 --step_size 256`
`python src/preprocess.py --stain reti --patch_size 224 --step_size 112 --output_dir data/processed_grading`
`python src/preprocess.py --stain he --patch_size 512 --step_size 256 --output_dir data/processed_subtype`

`python src/run_janitor.py \
    --task subtype \
    --input_dir data/processed_subtype_clean \
    --threshold 0.80`

`python src/run_janitor.py \
    --task grading \
    --input_dir data/processed_grading_clean \
    --threshold 0.80`