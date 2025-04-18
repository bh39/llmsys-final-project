#!/bin/bash

# Lists of cache sizes and thresholds to test
CACHE_SIZES=(10 20 50 100 200)
CACHE_THRESHOLDS=(1 2 4 6 8 10)


# Base config and output paths
CONFIG="../config/mmlu-3.yaml"
POLICY="LFU"
BASE_OUTDIR="LFU"

mkdir -p ${BASE_OUTDIR}

# Loop through all combinations of cache size and threshold
for size in "${CACHE_SIZES[@]}"; do
  for threshold in "${CACHE_THRESHOLDS[@]}"; do
    OUTDIR="${BASE_OUTDIR}/cache_size_${size}+threshold_${threshold}+policy_${POLICY}"
    
    echo "Running with cache size: ${size}, threshold: ${threshold}, output: ${OUTDIR}"
        
    # Call the Python script with the appropriate parameters
    python3 eval_mmlu.py \
      --config "${CONFIG}" \
      --outdir "${OUTDIR}" \
      --threshold ${threshold} \
      --cachesize ${size} \
      --cachepolicy "${POLICY}"
  done
done

echo "All evaluations completed"