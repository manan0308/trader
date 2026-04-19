#!/bin/zsh
set -euo pipefail

cd "/Users/mananagarwal/Desktop/2nd brain/plant to image/trader"

/Users/mananagarwal/miniconda3/bin/python -u research/alpha_v22_iterative_search.py \
  --samples-per-cycle 1 \
  --cycles 1 \
  --top-k 10 \
  --sleep-seconds 0
