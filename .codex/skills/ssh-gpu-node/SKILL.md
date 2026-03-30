---
name: ssh-gpu-node
description: Allocate an interactive GPU node for this lora repository over SSH. Use this skill when the user asks to get a GPU shell, request an interactive Slurm GPU allocation, SSH into a GPU node for this project, or generate/run the cluster command with this repo's defaults.
---

# SSH GPU Node

## Overview

Use this skill to allocate a single-GPU interactive shell for this repository on the remote cluster.

Default values for this project:

- Host: `xxphpc-jimin`
- Remote directory: `~/yizheng/lora`
- Partition: `gpu4090`
- QoS: `4gpus`
- CPUs per GPU: `4`

## Default Command

```bash
ssh -t xxphpc-jimin "cd ~/yizheng/lora; srun- --partition gpu4090 --qos 4gpus --gres=gpu:1 --cpus-per-gpu=4 --pty /bin/bash"
```

## Workflow

1. If the user wants the command only, provide the exact SSH command with these defaults unless they requested overrides.
2. If the user wants the allocation executed from the terminal, run `./.codex/skills/ssh-gpu-node/scripts/ssh_gpu_node.sh --run`.
3. Only change `host`, `remote_dir`, `partition`, `qos`, or `cpus` when the user explicitly asks for different values.
4. Keep `--gres=gpu:1` unless the user explicitly asks for a different GPU count.

## Script

Use `./.codex/skills/ssh-gpu-node/scripts/ssh_gpu_node.sh` for deterministic command generation.

- No flags: print the fully rendered SSH command
- `--run`: execute the SSH command
- `--host <host>`: override the SSH host
- `--remote-dir <path>`: override the remote working directory
- `--partition <name>`: override the Slurm partition
- `--qos <name>`: override the Slurm QoS
- `--cpus <count>`: override `--cpus-per-gpu`

Example:

```bash
./.codex/skills/ssh-gpu-node/scripts/ssh_gpu_node.sh
./.codex/skills/ssh-gpu-node/scripts/ssh_gpu_node.sh --run
```
