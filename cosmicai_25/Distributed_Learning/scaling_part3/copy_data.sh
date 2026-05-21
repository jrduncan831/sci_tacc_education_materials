SOURCE_TAR=$1
DEST_DIR=$2



FULL_CMD="mkdir -p $DEST_DIR; cp -r $SOURCE_TAR $DEST_DIR/. ; "


NODEFILE=/tmp/nodelist
scontrol show hostnames $SLURM_NODELIST > $NODEFILE

if [[ -z "${NODEFILE}" ]]; then
    RANKS=$HOSTNAME
else
    RANKS=$(tr '\n' ' ' < $NODEFILE)
fi

echo "Command: $FULL_CMD"

# Launch execute the command on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait