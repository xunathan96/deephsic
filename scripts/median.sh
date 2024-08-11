#!/bin/bash
cd ../src

# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval/riab/median --n-samples 500
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval/riab/median --n-samples 1000
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval/riab/median --n-samples 2000
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval/riab/median --n-samples 3000
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval/riab/median --n-samples 4000
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval/riab/median --n-samples 5000


python eval_hsic_median.py --dataset Sinusoid --save-dir exp/eval/sinusoid/median --n-samples 500 --n-permutations 100

cd -