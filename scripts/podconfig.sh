# Pod connection config — sourced by deploy/sync/tunnel scripts.
#
# RunPod pods get NEW SSH ports on every restart. Update POD_PORT below when
# the current pod restarts; all scripts that source this file pick up the new
# value. Everything else is stable (IP, key path, on-pod workspace path).
#
# Override any value via env: `POD_PORT=12345 ./deploy_to_pod.sh`

POD_HOST="${POD_HOST:-root@213.173.103.195}"
POD_PORT="${POD_PORT:-29770}"
POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519_runpod}"
POD_DIR="${POD_DIR:-/workspace/nimcp}"
POD_PY_SITE="${POD_PY_SITE:-/usr/local/lib/python3.12/dist-packages}"
POD_SOCKET="${POD_SOCKET:-/var/run/athena/brain.sock}"
POD_LOGDIR="${POD_LOGDIR:-/var/log/athena}"
POD_CKPT="${POD_CKPT:-/workspace/nimcp/checkpoints/athena/athena_immersive.bin}"

# Common ssh/scp commands with all the right flags — source this file and use
# directly as `$POD_SSH "remote command"` or `$POD_SCP local $POD_HOST:path`.
POD_SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i $POD_KEY -p $POD_PORT $POD_HOST"
POD_SCP="scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i $POD_KEY -P $POD_PORT"
