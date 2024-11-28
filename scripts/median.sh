#!/bin/bash
cd ../src

# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval-final/riab-final/logs/agg --n-samples 500
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval-final/riab-final/logs/agg --n-samples 1000
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval-final/riab-final/logs/agg --n-samples 2000
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval-final/riab-final/logs/agg --n-samples 3000
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval-final/riab-final/logs/agg --n-samples 4000
# python eval_hsic_median.py --dataset RatInABox --save-dir exp/eval-final/riab-final/logs/agg --n-samples 5000 --n-permutations 100


# python eval_hsic_median.py --dataset Sinusoid --save-dir exp/eval/sinusoid/median --n-samples 100 --n-permutations 100
# python eval_hsic_median.py --dataset Sinusoid --save-dir exp/eval/sinusoid/median --n-samples 200 --n-permutations 100
# python eval_hsic_median.py --dataset Sinusoid --save-dir exp/eval/sinusoid/median --n-samples 500 --n-permutations 100

# python eval_hsic_median.py --dataset HDGM-4 --save-dir exp/eval-final/hdgm-final/logs/agg --n-samples 100 --n-permutations 100
# python eval_hsic_median.py --dataset HDGM-4 --save-dir exp/eval-final/hdgm-final/logs/agg --n-samples 200 --n-permutations 100
# python eval_hsic_median.py --dataset HDGM-4 --save-dir exp/eval-final/hdgm-final/logs/agg --n-samples 500 --n-permutations 100
# python eval_hsic_median.py --dataset HDGM-4 --save-dir exp/eval-final/hdgm-final/logs/agg --n-samples 1000 --n-permutations 100
# python eval_hsic_median.py --dataset HDGM-4 --save-dir exp/eval-final/hdgm-final/logs/agg --n-samples 2000 --n-permutations 100

python eval_hsic_median.py --dataset Mice --save-dir exp/eval/wine/median --n-samples 1000 --n-permutations 100

cd -