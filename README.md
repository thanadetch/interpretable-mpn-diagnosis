# Example Commands to Train, Evaluate, and Explain Models

## Training Commands
### Resize Data Mode
`python src/train.py --task classification --data_mode resize --model resnet18 --epochs 50`

`python src/train.py --task grading --data_mode resize --model resnet18 --epochs 50`

### Patch Data Mode
`python src/train.py --task classification --data_mode subtype_patch_clean --model resnet18 --epochs 50`
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
`python src/evaluate.py --checkpoint experiments/classification_densenet121_subtype_patch_clean_20260121_122850/best_model.pth --data_mode subtype_patch_clean  --aggregation mean --level patient`
`python src/evaluate.py --checkpoint experiments/classification_resnet18_subtype_patch_clean_20260207_143722_ce_weighted_full/best_model.pth --data_mode subtype_patch_clean  --aggregation mean --level patient`

`python src/explain.py --checkpoint experiments/classification_resnet18_patch_20260101_220104/best_model.pth --data_mode patch --num_samples 10`
`python src/explain.py --checkpoint experiments/classification_densenet121_subtype_patch_clean_20260121_122850/best_model.pth --data_mode subtype_patch_clean --num_samples 50`

### Resize Data Mode
`python src/evaluate.py --checkpoint experiments/classification_resnet18_resize_20260101_210945/best_model.pth --data_mode resize`
`python src/evaluate.py --checkpoint experiments/classification_resnet18_resize_20260112_222224/best_model.pth --data_mode resize`

`python src/explain.py --checkpoint experiments/classification_resnet18_resize_20260101_210945/best_model.pth --data_mode resize --num_samples 10`
`python src/explain.py --checkpoint experiments/classification_resnet18_patch_20260112_151612/best_model.pth --data_mode patch --num_samples 30`
`python src/explain.py --checkpoint experiments/classification_resnet18_subtype_patch_clean_20260207_143722_ce_weighted_full/best_model.pth --data_mode subtype_patch_clean --num_samples 50`

### *Grading Task*
### Patch Data Mode
`python src/evaluate.py --checkpoint experiments/grading_resnet18_patch_20260108_023815/best_model.pth --data_mode patch`

`python src/evaluate.py --checkpoint experiments/grading_densenet121_grading_patch_clean_20260121_131139/best_model.pth --data_mode grading_patch_clean --aggregation mean --level image`

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
`python src/data/preprocess.py --patch_size 512 --step_size 256`
`python src/data/preprocess.py --stain reti --patch_size 224 --step_size 112 --output_dir data/processed_grading`
`python src/data/preprocess.py --stain he --patch_size 512 --step_size 256 --output_dir data/processed_subtype`
`python src/data/preprocess.py --stain he --patch_size 224 --step_size 112 --output_dir data/processed_subtype`

`python src/janitor/run_janitor.py --task grading --model_path experiments/janitor_grading_20260202_143052/janitor_model_grading.pth`

`python src/janitor/run_janitor.py \
    --task subtype \
    --input_dir data/processed_subtype_clean \
    --threshold 0.80`

`python src/janitor/run_janitor.py \
    --task grading \
    --input_dir data/processed_grading_clean \
    --threshold 0.80`

`python src/tools/data_stats.py`


# Classification task 
`python src/tools/find_best_seed.py --task classification --data_mode subtype_patch_clean` 

# Grading task
`python src/tools/find_best_seed.py --task grading --data_mode grading_patch_clean`


`python src/train_slide_encoder.py --features_dir data/features_titan --epochs 50 --lr 1e-4 --seed 42`

`python src/tools/visualize_features.py --backbone titan`

`python src/visualize_heatmap.py \
    --mil_checkpoint experiments/simple_mil_titan_20260218_042451/best_simple_titan.pth \
    --patient_dir "data/processed_subtype/PMF/PMF4 G2" \
    --image_id 1`

`python src/tools/visualize_test_set.py \
    --mil_checkpoint experiments/simple_mil_titan_20260218_042451/best_simple_titan.pth`