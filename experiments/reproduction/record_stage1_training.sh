#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   cd woCoefNet/Codes
#   bash ../../experiments/reproduction/record_stage1_training.sh -- \
#     python train.py --gpu 0 --batch_size 4 --max_epoch 100 \
#     --train_path /absolute/path/training/ --test_path /absolute/path/testing/
#
# Set SAMPLE_SECONDS=5 to change the nvidia-smi sampling interval.

if [[ "${1:-}" != "--" || "$#" -lt 2 ]]; then
  echo "Usage: $0 -- COMMAND [ARG ...]" >&2
  exit 2
fi
shift

project_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Run this script from inside the RopStitch Git repository." >&2
  exit 2
}

if [[ "$(pwd -P)" != "${project_root}/woCoefNet/Codes" ]]; then
  echo "WARNING: official train.py derives MODEL_DIR from the current working directory." >&2
  echo "Recommended working directory: ${project_root}/woCoefNet/Codes" >&2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${project_root}/experiments/reproduction/stage1/runs/${timestamp}"
mkdir -p "${run_dir}/checkpoints_before" "${run_dir}/checkpoints_after"

train_log="${run_dir}/train.log"
gpu_log="${run_dir}/gpu.csv"
metadata="${run_dir}/metadata.txt"
exit_file="${run_dir}/exit_status.txt"
sample_seconds="${SAMPLE_SECONDS:-10}"

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

{
  echo "run_dir=${run_dir}"
  echo "project_root=${project_root}"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "working_directory=$(pwd)"
  echo "git_commit=$(git -C "${project_root}" rev-parse HEAD)"
  echo "git_branch=$(git -C "${project_root}" branch --show-current)"
  echo "conda_environment=${CONDA_DEFAULT_ENV:-<none>}"
  echo "python_executable=$(command -v python || true)"
  echo "sample_seconds=${sample_seconds}"
  printf 'command='
  quote_command "$@"
  echo
  echo "===== git status ====="
  git -C "${project_root}" status --short --branch
  echo
  echo "===== git diff --stat ====="
  git -C "${project_root}" diff --stat
  echo
  echo "===== git diff ====="
  git -C "${project_root}" diff
  echo
  echo "===== Python packages ====="
  python - <<'PY'
import os
import platform
import sys

print("OS:", platform.platform())
print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "<none>"))
for module_name, display_name in (
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("skimage", "scikit-image"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
):
    try:
        module = __import__(module_name)
        print(f"{display_name}:", getattr(module, "__version__", "<unknown>"))
    except Exception as exc:
        print(f"{display_name} import error:", repr(exc))

try:
    import torch
    print("CUDA build:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        print(f"GPU {index}: {props.name}")
        print(f"GPU {index} total bytes: {props.total_memory}")
        print(f"GPU {index} free bytes at start: {free}")
except Exception as exc:
    print("CUDA inspection error:", repr(exc))
PY
  echo
  echo "===== nvidia-smi at start ====="
  nvidia-smi 2>&1 || true
} >"${metadata}" 2>&1

# Snapshot checkpoint names and sizes. This does not copy potentially large files.
find "${project_root}/woCoefNet/model_homo" \
  -maxdepth 1 -type f -name '*.pth' -printf '%f,%s,%TY-%Tm-%TdT%TH:%TM:%TS\n' \
  2>/dev/null | sort >"${run_dir}/checkpoints_before/list.csv"

echo "timestamp,index,name,uuid,memory.total_MiB,memory.used_MiB,memory.free_MiB,utilization.gpu_percent,temperature_C,power_W" >"${gpu_log}"
monitor_gpu() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader,nounits >>"${gpu_log}" 2>>"${run_dir}/gpu_monitor_errors.log" || true
    sleep "${sample_seconds}"
  done
}

monitor_gpu &
monitor_pid=$!

cleanup_monitor() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup_monitor EXIT INT TERM

echo "Training record directory: ${run_dir}"
echo "Command: $(quote_command "$@")"
echo "Start: $(date --iso-8601=seconds)"

set +e
"$@" > >(tee -a "${train_log}") 2> >(tee -a "${train_log}" >&2)
train_status=$?
set -e

cleanup_monitor
trap - EXIT INT TERM

find "${project_root}/woCoefNet/model_homo" \
  -maxdepth 1 -type f -name '*.pth' -printf '%f,%s,%TY-%Tm-%TdT%TH:%TM:%TS\n' \
  2>/dev/null | sort >"${run_dir}/checkpoints_after/list.csv"

{
  echo "exit_code=${train_status}"
  echo "end_time=$(date --iso-8601=seconds)"
  echo "last_log_line=$(tail -n 1 "${train_log}" 2>/dev/null || true)"
} >"${exit_file}"

{
  echo
  echo "===== nvidia-smi at end ====="
  nvidia-smi 2>&1 || true
  echo
  echo "end_time=$(date --iso-8601=seconds)"
  echo "exit_code=${train_status}"
} >>"${metadata}"

echo "End: $(date --iso-8601=seconds)"
echo "Exit code: ${train_status}"
echo "Records: ${run_dir}"
exit "${train_status}"
