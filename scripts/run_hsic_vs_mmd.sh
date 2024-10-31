#!/bin/bash
cd ../src

run=hsic_vs_mmd
python eval_hsic_vs_mmd.py --dataset Sinusoid.1f.1d \
                           --save-dir exp/eval/sinusoid/$run \
                           --method mmd \
                           --n-tests 100 \
                           --n-shuffles 5 \
                           --n-permutations 200 \
                           --n-samples 50 \


cd -