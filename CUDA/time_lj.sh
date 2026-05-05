#!/bin/bash

SIZES=(1000 2000 4000 8000)
THREADS_LIST=(1 64 256 4096 8192)
RUNS=5
N_STEPS=10

for threads in "${THREADS_LIST[@]}"; do
    for N in "${SIZES[@]}"; do
        for run in $(seq 1 $RUNS); do
            ./run-lj.sh "$N" "$threads" "$run" "$N_STEPS"
        done
    done
done
