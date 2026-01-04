#!/bin/bash
set -euo pipefail

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# PROFILE=4090_2x bash speedrun.sh
# 2) Example launch in a tmux session (because the run takes ~4 hours):
# bash speedrun.sh 2>&1 | tee speedrun.log
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun PROFILE=4090_2x bash speedrun.sh 2>&1 | tee speedrun.log

# Optional hardware profiles (can be overridden by CLI flags later if needed)
PROFILE="${PROFILE:-}"
# Defaults (same as original script: 8x H100 style)
GPU_COUNT=8
DEVICE_BATCH_SIZE=""
SFT_DEVICE_BATCH_SIZE=""
TOTAL_BATCH_SIZE=""
case "$PROFILE" in
    "") : ;; # no profile, keep defaults
    4090_2x)
        GPU_COUNT=2
        DEVICE_BATCH_SIZE=4
        SFT_DEVICE_BATCH_SIZE=1
        TOTAL_BATCH_SIZE=524288  #  524288 = 16K * 32,  393216 = 16K * 24, 491520 = 16K * 30
        ;;
    4090_8x)
        GPU_COUNT=8
        DEVICE_BATCH_SIZE=4
        SFT_DEVICE_BATCH_SIZE=1
        TOTAL_BATCH_SIZE=524288
        ;;
    
    5090_2x)
        GPU_COUNT=2
        DEVICE_BATCH_SIZE=6
        SFT_DEVICE_BATCH_SIZE=2
        TOTAL_BATCH_SIZE=393216
        ;;
    5090_8x)
        GPU_COUNT=8
        DEVICE_BATCH_SIZE=6
        SFT_DEVICE_BATCH_SIZE=2
        TOTAL_BATCH_SIZE=393216
        ;;

    H100_2x)
        GPU_COUNT=2
        DEVICE_BATCH_SIZE=32
        SFT_DEVICE_BATCH_SIZE=4
        TOTAL_BATCH_SIZE=524288
        ;;
    H100_8x)
        GPU_COUNT=8
        DEVICE_BATCH_SIZE=32
        SFT_DEVICE_BATCH_SIZE=4
        TOTAL_BATCH_SIZE=524288
        ;;
    *)
        echo "Unknown PROFILE: $PROFILE"
        exit 1
        ;;
esac

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Derived CLI args from profile overrides
DEVICE_BATCH_ARG=""
if [ -n "$DEVICE_BATCH_SIZE" ]; then
    DEVICE_BATCH_ARG="--device_batch_size=$DEVICE_BATCH_SIZE"
fi
SFT_DEVICE_BATCH_ARG=""
if [ -n "$SFT_DEVICE_BATCH_SIZE" ]; then
    SFT_DEVICE_BATCH_ARG="--sft_device_batch_size=$SFT_DEVICE_BATCH_SIZE"
fi
CHAT_EVAL_ARG=""
if [ -n "$DEVICE_BATCH_SIZE" ]; then
    CHAT_EVAL_ARG="--batch-size=$DEVICE_BATCH_SIZE"
fi


TOTAL_BATCH_ARG=""
if [ -n "$TOTAL_BATCH_SIZE" ]; then
    TOTAL_BATCH_ARG="--total_batch_size=$TOTAL_BATCH_SIZE"
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo if not already installed
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    source "$HOME/.cargo/env"
    echo "Rust/Cargo already installed, skipping install"
fi

# Build the rustbpe Tokenizer (skip if already importable in current venv)
if python - <<'PY'
import importlib.util
exit(0 if importlib.util.find_spec("rustbpe") else 1)
PY
then
    echo "rustbpe already built, skipping maturin develop"
else
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
if [ -f "$TOKENIZER_DIR/tokenizer.json" ] && [ -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "Tokenizer found at $TOKENIZER_DIR, skipping tokenizer training/eval"
else
    # train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
    python -m scripts.tok_train --max_chars=2000000000
    # evaluate the tokenizer (report compression ratio etc.)
    python -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
NPROC_PER_NODE=$GPU_COUNT

# pretrain the d20 model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN $DEVICE_BATCH_ARG $TOTAL_BATCH_ARG
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss $DEVICE_BATCH_ARG $TOTAL_BATCH_ARG
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN $DEVICE_BATCH_ARG $TOTAL_BATCH_ARG
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid $CHAT_EVAL_ARG

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN $SFT_DEVICE_BATCH_ARG
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft $CHAT_EVAL_ARG

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
