#!/bin/bash
# Preprocess attention masks for training datasets

# Activate conda environment if needed
# conda activate pfm

echo "Preprocessing attention masks for unplug_charger task..."

# Process both train and valid splits
# Note: Not specifying --n-points to save full masks (dataset will sample)
python scripts/preprocess_attention_masks.py \
    --task unplug_charger \
    --split both \
    --object-ids 31 34 35 92 \
    --distance-threshold 0.01 \
    --gripper-radius 0.05 \
    --overwrite

echo "Preprocessing complete!"