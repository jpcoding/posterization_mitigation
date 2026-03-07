#!/usr/bin/env bash

# Benchmark configuration for script/run_benchmarks.sh.
# Update the dataset paths on a new machine. Executable paths stay relative to the repo.

CPU_THREADS=8
CPU_INDEX_MODE=packed
BENCHMARK_ENABLE_GPU=auto
GPU_JFA_LEVEL=1

NYX_INPUT="/home/jp/data/nyx_512x512x512/velocity_x.f32"
NYX_REL_EB=0.01
NYX_CPU_DIMS="512 512 512"
NYX_GPU_DIMS="512 512 512"

HURRICANE_INPUT="/home/jp/data/hurricane_100x500x500/Uf48.bin.f32"
HURRICANE_REL_EB=0.01
HURRICANE_CPU_DIMS="100 500 500"
HURRICANE_GPU_DIMS="500 500 100"
