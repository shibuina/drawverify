#!/bin/bash
#SBATCH --account=sunjunresearch
#SBATCH --partition=sunjunresearch
#SBATCH --qos=sunjunresearch-priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/d/doducanh.2026/drawverify/logs/%x-%A_%a.log
# CS707 report runs on origami (arena-gpu01 was busy), lab node hexagon (H100 NVL), 2 GPUs at most:
#   sbatch -J dv --array=0-4%2 job.sh       # the five runs below, by array index
#   sbatch -J dv job.sh seed full4ans 1 --answer-only    # or one run by hand
# Env: ~/envs/lerobot (torch 2.11.0+cu130) + ~/drawverify/pydeps, the arena-gpu01 venv's versions (transformers 5.20.2).
# Requeue-safe: training resumes from its last epoch checkpoint, finished evals are skipped.
set -euo pipefail
cd ~/drawverify
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=~/drawverify/hf HF_HUB_OFFLINE=1
PY="env PYTHONPATH=$HOME/drawverify/pydeps $HOME/envs/lerobot/bin/python"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ $# -eq 0 ]; then
  RUNS=("seed full4ans 1 --answer-only" "seed full4ans 2 --answer-only" "seed full4 1" "seed full4 2" "zeroshot")
  set -- ${RUNS[$SLURM_ARRAY_TASK_ID]}
fi
echo "run: $*"
echo "job $SLURM_JOB_ID on $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) start $(date -Is)"
# peak GPU memory: sampled every 2 s for the whole job, max printed at the end
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > logs/mem-$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID:-0}.txt &
MEM=$!; trap 'kill $MEM; echo "peak GPU memory MiB: $(sort -n logs/mem-$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID:-0}.txt | tail -1)"; echo "end $(date -Is)"' EXIT
E() { local out=$1; shift; [ -s "$out" ] && return; $PY code/eval/eval_vlm.py --batch-size 8 "$@" --out "$out.part" && mv "$out.part" "$out"; }
O=runs/eval/cluster; mkdir -p $O
case $1 in
seed)
  name=$2 seed=$3; shift 3; M=Qwen/Qwen3-VL-2B-Instruct; R=runs/2b-$name-s$seed
  if [ ! -s $R/adapter_model.safetensors ]; then
    t0=$SECONDS
    $PY code/training/train.py --model $M --data data/full4 --seed $seed --output-dir $R "$@"
    echo "train wall s: $((SECONDS - t0))"
  fi
  E $O/$name-s${seed}_human.jsonl   --model $M --adapter $R --data data/human_ui3   --split test
  E $O/$name-s${seed}_cal.jsonl     --model $M --adapter $R --data data/human_train --split validation
  E $O/$name-s${seed}_l90val.jsonl  --model $M --adapter $R --data data/l90v4       --split validation
  E $O/$name-s${seed}_gentest.jsonl --model $M --adapter $R --data data/full4       --split test ;;
zeroshot)
  for m in 4B; do
    E $O/zeroshot-${m}_human.jsonl  --model Qwen/Qwen3-VL-$m-Instruct --data data/human_ui3 --split test
    E $O/zeroshot-${m}_l90val.jsonl --model Qwen/Qwen3-VL-$m-Instruct --data data/l90v4   --split validation
  done
  # pixel-geometry baseline (CPU, LightGBM on circle/arrow positions; never reads the caption)
  ls $O/geometry/*.jsonl >/dev/null 2>&1 || $PY code/eval/geometry_baseline.py --fit gen=data/full4:train --fit hum=data/human_train:train \
      --fit gen+hum=data/full4:train,data/human_train:train \
      --test human=data/human_ui3:test --test l90=data/l90v4:validation --out-dir $O/geometry ;;
datasize)
  frac=$2; tag=${frac/./p}; M=Qwen/Qwen3-VL-2B-Instruct; R=runs/2b-full4ans-f$tag
  if [ ! -s $R/adapter_model.safetensors ]; then
    t0=$SECONDS
    $PY code/training/train.py --model $M --data data/full4 --seed 0 --answer-only --subsample-frac "$frac" --output-dir $R
    echo "train wall s: $((SECONDS - t0))"
  fi
  E $O/full4ans-f${tag}_human.jsonl   --model $M --adapter $R --data data/human_ui3   --split test
  E $O/full4ans-f${tag}_l90val.jsonl  --model $M --adapter $R --data data/l90v4       --split validation
  E $O/full4ans-f${tag}_gentest.jsonl --model $M --adapter $R --data data/full4       --split test ;;
esac
echo ALLDONE
