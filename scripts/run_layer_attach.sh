#!/bin/bash
# Layer-attachment runs (Step 2 follow-up): concept bank attaches to
# layer2 (RF 0.99s) / layer3 (RF 2.11s) instead of layer4 (RF 4.35s).
# 6 runs, serial on one GPU, wait-mode scheduling for the shared cluster:
# poll until some GPU has util < 25% AND > 30GB free, then grab it.
# Never touches other users' processes.
# Usage: bash scripts/run_layer_attach.sh

set -u
LOG_DIR=results/layer_attach_logs
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/queue.log"

CONFIGS=(
  configs/hmacd_k32_lc0.1_ld0.05_l2_seed42.yaml
  configs/hmacd_k32_lc0.1_ld0.05_l2_seed43.yaml
  configs/hmacd_k32_lc0.1_ld0.05_l2_seed44.yaml
  configs/hmacd_k32_lc0.1_ld0.05_l3_seed42.yaml
  configs/hmacd_k32_lc0.1_ld0.05_l3_seed43.yaml
  configs/hmacd_k32_lc0.1_ld0.05_l3_seed44.yaml
)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"; }

wait_for_gpu() {
  # returns a free GPU index, polling every 60s
  while true; do
    for gpu in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      read util mem_used mem_total < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits -i "$gpu" | tr ',' ' ')
      free=$((mem_total - mem_used))
      if [ "$util" -lt 25 ] && [ "$free" -gt 30720 ]; then
        echo "$gpu"
        return 0
      fi
    done
    sleep 60
  done
}

log "queue start: ${#CONFIGS[@]} runs, wait mode"
GPU=$(wait_for_gpu)
log "acquired GPU $GPU"

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  log="run log: $LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] start $name" | tee -a "$QUEUE_LOG"
  if python scripts/train_hmacd.py --config "$cfg" --gpu_ids "$GPU" --no-wandb \
      > "$LOG_DIR/${name}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK    $name" | tee -a "$QUEUE_LOG"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see ${name}.log)" | tee -a "$QUEUE_LOG"
    exit 1
  fi
done

echo "[$(date '+%F %T')] queue done, all OK" | tee -a "$QUEUE_LOG"
