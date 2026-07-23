#!/bin/bash
#SBATCH --job-name=qwen_pope
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/pope_qwen/slurm_%j.log
#SBATCH --error=outputs/pope_qwen/slurm_%j.log

# Activate environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate qwen_cd

cd /mnt/home2/home/ankur_d/sm/cd-rethinking

# Log file with timestamp
LOG_FILE="outputs/pope_qwen/logs/run_qwen_pope_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/pope_qwen/logs

# Redirect all stdout and stderr to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== Qwen2.5-VL-7B POPE run started at $(date) ====="
echo "Logging to: $LOG_FILE"
echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Slurm assigns 1 GPU (index 0 in the job's isolated CUDA space).
# Patch pope_qwen.sh to use only that single GPU with no sharding.
sed 's/^GPUS=.*/GPUS=(0)/' scripts/pope_qwen.sh > /tmp/pope_qwen_slurm_$$.sh
chmod +x /tmp/pope_qwen_slurm_$$.sh

echo "----- Running Qwen2.5-VL-7B-Instruct POPE table (all methods, resumable) -----"
bash /tmp/pope_qwen_slurm_$$.sh

echo ""
echo "===== Qwen2.5-VL-7B POPE run finished at $(date) ====="

rm -f /tmp/pope_qwen_slurm_$$.sh
