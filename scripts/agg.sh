#!/bin/bash
cd ../src

# savedir=exp/eval-final/riab-final/logs/agg
# python eval_hsic_agg.py --dataset RatInABox --save-dir $savedir --n-samples 100
# python eval_hsic_agg.py --dataset RatInABox --save-dir $savedir --n-samples 200
# python eval_hsic_agg.py --dataset RatInABox --save-dir $savedir --n-samples 500
# python eval_hsic_agg.py --dataset RatInABox --save-dir $savedir --n-samples 1000
# python eval_hsic_agg.py --dataset RatInABox --save-dir $savedir --n-samples 2000
# python eval_hsic_agg.py --dataset RatInABox --save-dir $savedir --n-samples 3000
# python eval_hsic_agg.py --dataset RatInABox --save-dir $savedir --n-samples 4000
# python eval_hsic_agg.py --dataset RatInABox --save-dir $savedir --n-samples 5000

savedir=exp/eval/wine/agg
python eval_hsic_agg.py --dataset Wine --save-dir $savedir --n-samples 1200
python eval_hsic_agg.py --dataset Wine --save-dir $savedir --n-samples 1300


cd -