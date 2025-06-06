# Attention Mask Preprocessing Guide

## Overview
This guide explains how to preprocess attention masks for the PointFlowMatch framework with supervised 3D spatial attention. Preprocessing masks significantly speeds up training by moving expensive computations from training time to a one-time preprocessing step.

## Why Preprocess?
- **Speed**: Eliminates expensive mask generation during training (was happening thousands of times)
- **Consistency**: Ensures same masks are used across training runs
- **Memory**: Pre-computed masks are more efficient than storing full segmentation data
- **Flexibility**: Can still fall back to on-the-fly generation if needed

## Prerequisites
1. Collected demos with segmentation data using:
   ```bash
   python scripts/collect_demos_with_segmentation.py
   ```
   This creates `train_segmented/` and `valid_segmented/` directories.

2. Identified object IDs for your task (e.g., for unplug_charger: [31, 34, 35, 92])

## Preprocessing Steps

### 1. Run Preprocessing Script
```bash
# For unplug_charger task (default)
bash bash/preprocess_masks.sh

# Or run directly with custom parameters
python scripts/preprocess_attention_masks.py \
    --task unplug_charger \
    --split both \
    --object-ids 31 34 35 92 \
    --distance-threshold 0.01 \
    --gripper-radius 0.05 \
    --n-points 4096
```

### 2. Verify Preprocessing
```bash
python scripts/test_preprocessed_masks.py
```

This will show:
- Whether attention masks were found
- Shape and coverage statistics
- Test loading through the dataset

## Parameters

### Object IDs
- Task-specific object IDs to track
- Default: `[31, 34, 35, 92]` (unplug_charger)
- Find IDs using: `python scripts/get_object_names.py`

### Distance Threshold
- Maximum distance (meters) from object points to be included in attention
- Default: `0.01` (1cm)
- Smaller = tighter attention around objects

### Gripper Radius
- Radius around gripper to always include in attention
- Default: `0.05` (5cm)
- Ensures gripper region is always attended to

### Number of Points
- Number of points to sample from point cloud
- Default: `4096`
- Must match training configuration

## Training with Preprocessed Masks

### 1. Use Attention Experiment Config
```bash
python scripts/train.py +experiment=pointflowmatch_so3_attention log_wandb=True
```

### 2. Dataset Configuration
The attention dataset (`dataset_pcd_attention.py`) automatically:
1. Checks for pre-computed masks first
2. Loads them if available (fast)
3. Falls back to on-the-fly generation if not found (slow)

### 3. Monitor Performance
With preprocessing, you should see:
- Faster data loading during training
- No warnings about generating masks on-the-fly
- Consistent attention masks across runs

## Troubleshooting

### "No preprocessed attention masks found"
- Ensure you ran preprocessing on the correct dataset
- Check that the data path contains `_segmented` suffix
- Verify the replay buffer has write permissions

### "Segmentation data not found"
- Make sure you collected demos with segmentation enabled
- Check that `segmentation_masks` and `camera_point_clouds` exist in the replay buffer

### Different attention coverage than expected
- Adjust `distance_threshold` (smaller = less coverage)
- Modify `gripper_radius` (larger = more coverage around gripper)
- Check object IDs are correct for your task

## Implementation Details

### Storage Format
- Attention masks stored in replay buffer under key `"attention_masks"`
- Shape: `(total_timesteps, n_points)`
- Dtype: `float32` (0.0 or 1.0 values)
- Chunked by timestep for efficient loading

### Sampling Consistency
- Preprocessing uses deterministic random seed: `episode_idx * 1000 + timestep`
- Dataset uses same seed to ensure consistent point sampling
- Critical for matching pre-computed masks to sampled points

### Memory Usage
- Attention masks: ~16KB per timestep (4096 points × 4 bytes)
- Much smaller than segmentation masks + camera point clouds
- Compressed using LZ4 by default

## Advanced Usage

### Custom Preprocessing
Modify `preprocess_attention_masks.py` to:
- Add different attention strategies
- Include velocity-based attention
- Weight attention by distance (soft attention)
- Add temporal smoothing

### Multiple Objects
For tasks with multiple objects:
1. Identify all relevant object IDs
2. Pass them all to `--object-ids`
3. The preprocessing will create union of all object regions

### Overwriting Existing Masks
```bash
python scripts/preprocess_attention_masks.py --task your_task --overwrite
```

Use this when:
- Changed object IDs
- Modified distance thresholds
- Updated preprocessing logic