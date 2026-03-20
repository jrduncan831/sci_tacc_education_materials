TAP_FUNCTIONS="/share/doc/slurm/tap_functions"
if [ -f ${TAP_FUNCTIONS} ]; then
    . ${TAP_FUNCTIONS}
else
    echo "TACC:"
    echo "TACC: ERROR - could not find TAP functions file: ${TAP_FUNCTIONS}"
    echo "TACC: ERROR - Please submit a consulting ticket at the TACC user portal"
    echo "TACC: ERROR - https://portal.tacc.utexas.edu/tacc-consulting/-/consult/tickets/create"
    echo "TACC:"
    echo "TACC: job $SLURM_JOB_ID execution finished at: `date`"
    exit 1
fi

NODE_HOSTNAME=$(hostname -s)
echo "TACC: running on node ${NODE_HOSTNAME}"

LOGIN_PORT_PHOENIX=$(tap_get_port)
echo "TACC: got login node jupyter port ${LOGIN_PORT}"

PHOENIX_PORT=5903
NUM_LOGINS=2
for i in $(seq ${NUM_LOGINS}); do
    ssh -q -f -g -N -R ${LOGIN_PORT_PHOENIX}:${NODE_HOSTNAME}:${PHOENIX_PORT} ilogin${i}
done
echo "Phoenix server accessible via browser at http://vista.tacc.utexas.edu:${LOGIN_PORT_PHOENIX}/ after launch"
