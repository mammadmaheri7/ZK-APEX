#!/bin/bash
#SBATCH --job-name=parallel_tmp
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Usage:
#   ./submit_tmp_parallel.sh gpu32 gpu33 gpu34 a

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 SERVER..."
  exit 1
fi

SERVERS=("$@")

# Ensure local log directory exists
mkdir -p parallel_logs

# --- Function to launch tmp.sh on a given node ---
run_on_node() {
  local server=$1
  local nodename
  nodename=$(basename "$server")

  echo "→ Launching tmp.sh on ${server}, logging to parallel_logs/logs_${nodename}.txt"

  ssh "$server" <<EOF
cd /vol/bitbucket/sc2124/selective_pruning_mmd/exp14-weight-pruning || exit 1
mkdir -p parallel_logs
stdbuf -oL -eL ./tmp.sh | tee -a parallel_logs/logs_${nodename}.txt
EOF
}

# --- Launch on all servers in parallel ---
for srv in "${SERVERS[@]}"; do
  run_on_node "$srv" &
done

wait
echo "✅ All nodes finished running tmp.sh"


