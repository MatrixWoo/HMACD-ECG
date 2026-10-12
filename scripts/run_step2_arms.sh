#!/bin/bash
# Step 2 evidence-span arms: serial queue on one GPU, stops on first failure.
# 6 runs: arm1 scatter-8 (control, same candidate count as arm2) x 3 seeds,
#         arm2 segment-4 x 3 seeds. Fixed budget: patience = epochs = 50.
# Usage: bash scripts/run_step2_arms.sh [gpu_id]   (default 3)

set -u
GPU=${1:-3}
LOG_DIR=results/step2_logs
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/queue.log"

CONFIGS=(
  configs/hmacd_k32_lc0.1_ld0.05_sc8_seed42.yaml
  configs/hmacd_k32_lc0.1_ld0.05_sc8_seed43.yaml
  configs/hmacd_k32_lc0.1_ld0.05_sc8_seed44.yaml
  configs/hmacd_k32_lc0.1_ld0.05_sg4_seed42.yaml
  configs/hmacd_k32_lc0.1_ld0.05_sg4_seed43.yaml
  configs/hmacd_k32_lc0.1_ld0.05_sg4_seed44.yaml
)

echo "[$(date '+%F %T')] queue start: ${#CONFIGS[@]} runs on GPU $GPU" | tee -a "$QUEUE_LOG"

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  log="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] start $name" | tee -a "$QUEUE_LOG"
  if python scripts/train_hmacd.py --config "$cfg" --gpu_ids "$GPU" --no-wandb > "$log" 2>&1; then
    echo "[$(date '+%F %T')] OK    $name" | tee -a "$QUEUE_LOG"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $log)" | tee -a "$QUEUE_LOG"
    exit 1
  fi
done

echo "[$(date '+%F %T')] queue done, all OK" | tee -a "$QUEUE_LOG"
