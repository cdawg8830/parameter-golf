#!/bin/bash
# RunPod 1xH100 setup and experiment script.
# Run this after SSH-ing into the pod (you'll land in /workspace).
# Each section is clearly labeled -- run them one at a time.

set -e

# ============================================================
# STEP 1: Clone and download data (~5 min for data download)
# ============================================================

cd /workspace
git clone https://github.com/cdawg8830/parameter-golf.git
cd parameter-golf

python3 data/cached_challenge_fineweb.py --variant sp1024

echo "Data ready. Training files:"
ls data/datasets/fineweb10B_sp1024/ | head -5
ls data/tokenizers/

# ============================================================
# STEP 2: Run the BASELINE (original 9-layer 512-dim model)
# This gives our 1-GPU reference score to beat.
# Expected: val_bpb ~1.23-1.25 (slightly worse than 8-GPU 1.2244)
# Runtime: ~10 min
# ============================================================

RUN_ID=baseline_1gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo "Baseline done. Check logs/baseline_1gpu*.txt for val_bpb"

# ============================================================
# STEP 3: Run the LOOPED model (3 unique blocks x 6 loops, 768 dim)
# This is our depth recurrence experiment.
# Expected: should beat baseline if hypothesis holds.
# Runtime: ~10-12 min (slightly slower per step due to wider dim)
# ============================================================

RUN_ID=looped_3x6_768 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_UNIQUE_LAYERS=3 \
LOOP_COUNT=6 \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo "Looped run done. Check logs/looped_3x6_768*.txt for val_bpb"

# ============================================================
# STEP 4: Budget-fill variant (mlp_mult=3, ~15M params)
# Only run if Step 3 beats Step 2.
# ============================================================

RUN_ID=looped_3x6_768_mlp3 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_UNIQUE_LAYERS=3 \
LOOP_COUNT=6 \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo "Budget-fill run done. Check logs/looped_3x6_768_mlp3*.txt for val_bpb"

# ============================================================
# COMPARE RESULTS
# ============================================================

echo ""
echo "=== RESULTS SUMMARY ==="
echo "--- Baseline ---"
grep "final_int8_zlib_roundtrip val_bpb" logs/baseline_1gpu*.txt 2>/dev/null | tail -1

echo "--- Looped 3x6 768 ---"
grep "final_int8_zlib_roundtrip val_bpb" logs/looped_3x6_768*.txt 2>/dev/null | tail -1

echo "--- Looped 3x6 768 mlp3 ---"
grep "final_int8_zlib_roundtrip val_bpb" logs/looped_3x6_768_mlp3*.txt 2>/dev/null | tail -1
