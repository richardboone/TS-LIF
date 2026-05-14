#!/bin/bash

# Default usage instructions
function usage() {
    echo "Usage: $0 [GPU_ID] [OUTPUT_DIR]"
    echo "  GPU_ID     : Integer index of target GPU (Default: 0)"
    echo "  OUTPUT_DIR : Destination for logs & checkpoints (Default: outputs/run_metr_la)"
    echo ""
    echo "Example: $0 6 outputs/my_experiment"
    exit 1
}

# Show usage if -h or --help passed
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Assign arguments with default fallbacks
GPU=${1:-0}
OUT_DIR=${2:-"outputs/run_metr_la"}

CONFIG_PATH="TS-LIF/exp/forecast/spikegru/local_spikegru_metr-la.yml"

echo "================================================"
echo "Starting METR-LA Experiment"
echo "  Target GPU     : $GPU"
echo "  Output Directory: $OUT_DIR"
echo "  Config Loaded   : $CONFIG_PATH"
echo "================================================"

# Ensure output directory structure exists
mkdir -p "$(dirname "$OUT_DIR")"

# Execute training inside Conda environment
CUDA_VISIBLE_DEVICES=$GPU conda run -n TS-LIF --no-capture-output python TS-LIF/SeqSNN/entry/tsforecast.py "$CONFIG_PATH" --runtime.output_dir "$OUT_DIR"
