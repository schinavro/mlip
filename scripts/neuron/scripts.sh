#!/bin/bash

source ~/.bashrc
conda activate simple

# Setup node list
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
master_node=${nodes_array[0]}
master_addr=$(srun --nodes=1 --ntasks=1 -w $master_node hostname --ip-address)
master_port=$((RANDOM%55535+10000))
worker_num=$(($SLURM_JOB_NUM_NODES))
gpus=$(echo $SLURM_JOB_GPUS | sed s/,/\\n/g)
gpu_array=( $gpus )
gpu_num=$(($SLURM_GPUS_ON_NODE))


# Loop over nodes and submit training tasks
for ((  node_idx=0; node_idx<$worker_num; node_idx++ )); do
    for ((  gpu_rank=0; gpu_rank<$gpu_num; gpu_rank++ )); do
          rank=$[$node_idx * $worker_num + $gpu_rank]
          node=${nodes_array[$node_idx]}
          gpu=${gpu_array[$gpu_rank]}
          echo "Submitting $node, with $rank, with gpu cuda:$gpu, at $master_addr:$master_port"
          echo "srun -N 1 -n 1 -w $node python reann_H2O.py \
                --master_port $master_port \
                --master_addr $master_addr \
                --backend 'nccl' \
                --node $node_idx \
                --device "cuda:$gpu" \
                --world_size 4 \
                --rank $rank"

          # Launch one SLURM task per node, and use torch distributed launch utility
          # to spawn training worker processes; one per GPU
          srun -N 1 -n 1 -w $node python reann_H2O.py \
                --master_port $master_port \
                --master_addr $master_addr \
                --backend 'nccl' \
                --node $node_idx \
                --device "cuda:$gpu" \
                --world_size 4 \
                --rank $rank &

          pids[${rank}]=$!
    done
done

# Wait for completion
for pid in ${pids[*]}; do
          wait $pid
done
