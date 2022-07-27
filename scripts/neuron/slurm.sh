#!/bin/sh
#SBATCH --job-name=Ge
#SBATCH -J Serial_gpu_job
#SBATCH -p cas_v100_2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1      # using 2 gpus per node
#SBATCH --comment pytorch #Application별 SBATCH 옵션 이름표 참고

source ~/.bashrc
conda activate simple

srun python /scratch/x2419a03/test/test.py Li

exit 0
