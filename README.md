# WildAnimalDetection

The code implements a trap-camera image classification system for Jasper Ridge (JR) Biological Preserve.
`best_model.pt` is needed but not included in the github due to its large size.

Author: Yifan Yu(@YifanYu5124), Yancheng Li(@lycyeah) and Tianpei Qian(qiantianpei)

### Dependencies

- Python 3.6 
- PyTorch 0.3

### Files

- `predict.py`: main file for prediction
- `WildAnimalDataset.py`: dataloader

### Prediction

```
python predict.py --threshold 0.9
```

