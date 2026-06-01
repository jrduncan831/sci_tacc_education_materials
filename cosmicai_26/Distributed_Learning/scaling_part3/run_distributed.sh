#!/bin/bash
LOCAL_RANK=$SLURM_PROCID

NODEFILE=/tmp/hostfile
#scontrol show hostnames  > $NODEFILE
if [[ -z "${NODEFILE}" ]]; then
    RANKS=$NODEFILE
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi


PRELOAD="/opt/apps/tacc-apptainer/1.4.1/bin/apptainer exec --nv --bind /run/user:/run/user /scratch/07980/sli4/containers/cosmicai_vista_v12.sif "
CMD="torchrun --nproc_per_node 1 --nnodes $NNODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MAIN_RANK --master_port=1234 torch_train_distributed.py $@"

FULL_CMD="$PRELOAD $CMD"
echo "Training command: $FULL_CMD"

eval $FULL_CMD