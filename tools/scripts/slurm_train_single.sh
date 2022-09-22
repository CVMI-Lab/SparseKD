#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=1
PY_ARGS=${@:3}

GPUS_PER_NODE=1
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train.py ${PY_ARGS}
