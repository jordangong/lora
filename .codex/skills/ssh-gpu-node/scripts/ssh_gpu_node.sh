#!/usr/bin/env bash

set -euo pipefail

HOST="xxphpc-jimin"
REMOTE_DIR="~/yizheng/lora"
PARTITION="gpu4090"
QOS="4gpus"
CPUS="4"
RUN=0

require_value() {
  if [[ $# -lt 2 || -z "${2}" ]]; then
    echo "Missing value for ${1}" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      require_value "$@"
      HOST="$2"
      shift 2
      ;;
    --remote-dir)
      require_value "$@"
      REMOTE_DIR="$2"
      shift 2
      ;;
    --partition)
      require_value "$@"
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      require_value "$@"
      QOS="$2"
      shift 2
      ;;
    --cpus)
      require_value "$@"
      CPUS="$2"
      shift 2
      ;;
    --run)
      RUN=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ssh_gpu_node.sh [options]

Print or run the SSH command that allocates an interactive GPU node.

Options:
  --host <host>             Override SSH host
  --remote-dir <path>       Override remote working directory
  --partition <name>        Override Slurm partition
  --qos <name>              Override Slurm QoS
  --cpus <count>            Override CPUs per GPU
  --run                     Execute the SSH command instead of printing it
  -h, --help                Show this help text
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

REMOTE_COMMAND="cd ${REMOTE_DIR}; srun- --partition ${PARTITION} --qos ${QOS} --gres=gpu:1 --cpus-per-gpu=${CPUS} --pty /bin/bash"

if [[ "${RUN}" -eq 1 ]]; then
  exec ssh -t "${HOST}" "${REMOTE_COMMAND}"
fi

printf 'ssh -t %s "%s"\n' "${HOST}" "${REMOTE_COMMAND}"
