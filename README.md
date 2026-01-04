`python src/evaluate.py --checkpoint experiments/classification_resnet18_patch_20260101_220104/best_model.pth --data_mode patch`
`python src/explain.py --checkpoint experiments/classification_resnet18_patch_20260101_220104/best_model.pth --data_mode patch --num_samples 10`

`python src/evaluate.py --checkpoint experiments/classification_resnet18_resize_20260101_210945/best_model.pth --data_mode resize`
`python src/explain.py --checkpoint experiments/classification_resnet18_resize_20260101_210945/best_model.pth --data_mode resize --num_samples 10`

`python src/evaluate.py --checkpoint experiments/grading_resnet18_patch_20260101_173308/best_model.pth --data_mode patch`
`python src/explain.py --checkpoint experiments/grading_resnet18_patch_20260101_173308/best_model.pth --data_mode patch --num_samples 10`

`python src/evaluate.py --checkpoint experiments/grading_resnet18_resize_20260101_164102/best_model.pth --data_mode resize`
`python src/explain.py --checkpoint experiments/grading_resnet18_resize_20260101_164102/best_model.pth --data_mode resize --num_samples 10`

