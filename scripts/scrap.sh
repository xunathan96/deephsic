#!/bin/bash
cd ../src

# config files
train_root=config/exp/train/
eval_root=config/exp/eval/
data_root=config/dataset/
model_root=config/model/
save_root=exp/pretrained/
log_root=exp/eval/

run=test
method=hsic
# dataset=penn_treebank.10000
dataset=penn_treebank.2000
# model=emb813x64x32.mlp128x64x32-mlp64x32x8
model=emb2893x64x32.mlp128x64x32-mlp64x32x8

python train.py --train-config $train_root/hsic/train.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/penn_treebank/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/penn_treebank/$dataset/hsic/$model/$run \
                --n-epochs 1000

python eval.py  --eval-config $eval_root/$method/eval.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/penn_treebank/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --pretrained-path $save_root/penn_treebank/$dataset/$method/$model/$run/best.pt \
                --log-dir $log_root/penn_treebank/$run \
                --n-samples 400


cd -
