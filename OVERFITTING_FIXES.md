# Overfitting Diagnosis & Fixes

## Problems Identified

### 1. **Insufficient Regularization** ⚠️
- **Issue**: Only 0.3 dropout before final layer, no intermediate dropout
- **Impact**: Model learns training data noise instead of generalizable patterns
- **Fix**: 
  - Added 2 dropout layers (0.5 and 0.4) in the classification head
  - Added **L2 regularization** (weight_decay=0.0005) to optimizer

### 2. **Weak Classification Head** ⚠️
- **Issue**: Single linear layer after dropout → too few parameters to learn meaningful features
- **Impact**: Model either underfits or overfits quickly
- **Fix**: 
  - Added intermediate layers: Linear(512) → ReLU → Linear(256) → ReLU → Linear(3)
  - Added BatchNormalization after each linear layer for stable training
  - Now: `2048 → 512 → 256 → 3` (better feature learning)

### 3. **Pretrained Model Not Fine-Tuned Properly** ⚠️
- **Issue**: All layers trainable from ImageNet weights → catastrophic forgetting
- **Impact**: Model "forgets" useful ImageNet features while overfitting to small dataset
- **Fix**: 
  - **Freeze early layers** (layer1, layer2) - keep ImageNet knowledge
  - **Train only layer3, layer4 + classification head** - task-specific learning
  - This is standard practice for transfer learning on small datasets

### 4. **Poor Learning Rate** ⚠️
- **Issue**: lr=0.0001 too small → slow convergence, need more epochs to find good minima
- **Impact**: Model stops improving before finding good solution
- **Fix**: 
  - Increased to lr=0.001 (10x) for better convergence
  - Combined with ReduceLROnPlateau scheduler for adaptive learning rates

### 5. **Duplicate Configuration** 🔴
- **Issue**: epochs and lr defined twice with conflicting values
- **Impact**: Confusion about which settings are used
- **Fix**: Cleaned up config file with single, clear definitions

### 6. **Limited Data Augmentation** ⚠️
- **Issue**: Only basic augmentation (flip, rotate, color jitter)
- **Impact**: Model sees too few variations of same data → overfits
- **Fix**: Enhanced augmentation pipeline:
  ```
  ✓ RandomVerticalFlip (p=0.2)
  ✓ RandomAffine (translation, scale)
  ✓ GaussianBlur (p=0.2)
  ✓ Increased ColorJitter probability to 0.6
  ```

### 7. **Insufficient Early Stopping Patience** ⚠️
- **Issue**: patience=4 may stop training too early
- **Impact**: Model doesn't converge to good solution
- **Fix**: Increased to patience=8 (more stable)

---

## Summary of Changes

### Model Architecture (`src/model.py`)
```python
# OLD: 2048 → Dropout(0.3) → 3 classes (too simple!)
# NEW: 2048 → 512(BN, ReLU, Dropout(0.5)) → 256(BN, ReLU, Dropout(0.4)) → 3 classes
```

**Benefits:**
- More capacity to learn features
- Stronger regularization with BatchNorm + Dropout
- Better gradient flow through layers

### Configuration (`configs/base.yaml`)
| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| lr | 0.0001 | 0.001 | Better convergence speed |
| epochs | 10-20 | 50 | More training with early stopping |
| weight_decay | - | 0.0005 | L2 regularization |
| patience | 4 | 8 | Stable training |
| freeze_backbone | - | true | Transfer learning best practice |

### Training (`src/train.py`)
- Added weight_decay to optimizer: `Adam(..., weight_decay=0.0005)`
- Enhanced data augmentation with 7 techniques
- Added parameter counting: shows trainable vs total params
- Added learning rate tracking in logs

---

## Expected Improvements

✅ **Better generalization** - model learns features, not noise
✅ **Reduced overfitting gap** - val_loss ≈ train_loss
✅ **Faster convergence** - 10x higher learning rate
✅ **Stable training** - batch norm + better architecture
✅ **Better fine-tuning** - frozen early layers prevent catastrophic forgetting

---

## How to Monitor Improvements

### Signs of Fixed Overfitting:
1. **Validation loss tracks training loss** (not diverging)
2. **Gap between train/val accuracy small** (< 5-10%)
3. **Validation metrics improving with training**
4. **Stable curves** (not jumping around)

### Track These Metrics:
```
Epoch 10: train_loss=0.6543 val_loss=0.6892  ✓ Good (small gap)
Epoch 20: train_loss=0.4123 val_loss=0.8921  ✗ Overfitting (large gap)
```

---

## Additional Tips

### If Still Overfitting:
1. **Reduce batch size** → more gradient updates (regularizing effect)
2. **Increase weight decay** → 0.001 or 0.0005
3. **Add more dropout** → increase all dropout rates by 0.1
4. **Collect more training data** → most effective solution
5. **Simpler model** → reduce 512→256 to 512→128

### If Underfitting:
1. **Increase learning rate** → try 0.002 or 0.0015
2. **Decrease weight decay** → try 0.0001
3. **More complex model** → increase hidden dimensions
4. **Decrease dropout** → try 0.3 and 0.2
5. **Longer training** → increase epochs

### Best Practice Checklist:
- [ ] Monitor both train and val loss/accuracy
- [ ] Use early stopping with patience=8+
- [ ] Freeze early layers in pretrained models
- [ ] Use data augmentation generously
- [ ] Add L2 regularization (weight decay)
- [ ] Use learning rate scheduler
- [ ] Log parameter counts (ensure trainable params reasonable)
- [ ] Keep validation set large and diverse

---

## Files Modified

1. **model.py** - Improved architecture with better regularization
2. **src/model.py** - Same improvements (duplicate file)
3. **configs/base.yaml** - Fixed config, added weight_decay & freeze_backbone
4. **src/train.py** - Added weight_decay, enhanced augmentation, better logging

---

## Next Steps

1. **Run training**: `python src/train.py --config configs/base.yaml`
2. **Monitor**: Watch for train/val loss to track together
3. **Adjust**: If needed, tweak weight_decay or dropout based on results
4. **Evaluate**: Test on test set after training stabilizes
