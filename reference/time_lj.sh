#!/bin/bash

SIZES=(1000 2000 4000 8000)
RUNS=5
N_STEPS=5000

for N in "${SIZES[@]}"; do
    for run in $(seq 1 $RUNS); do
        ./run-lj.sh "$N" "$run" "$N_STEPS"
    done
done