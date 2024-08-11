#!/bin/bash
#SBATCH --account=def-dsuth
#SBATCH --gpus-per-node=1       # Request 1 available GPU (--gpus-per-node=p100:1)
#SBATCH --mem=4000M             # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-48:00:00       # DD-HH:MM:SS
#SBATCH --job-name=sinusoid
#SBATCH --output=logs/%x/slurm-%j.out   # output file. %x is the job name, %N is the hostname, %j is the job id

PROJ_DIR=$project/deepkernel
VENV_DIR=$PROJ_DIR/myenv
SOURCE_DIR=$PROJ_DIR/src
SCRIPT_DIR=$PROJ_DIR/scripts
cd $SCRIPT_DIR

module load python/3.10 scipy-stack cuda cudnn
source $VENV_DIR/bin/activate

# config files
train_root=config/exp/train/
eval_root=config/exp/eval/
data_root=config/dataset/
model_root=config/model/
save_root=exp/pretrained/
log_root=exp/eval/


# dataset:testsizes
declare -A dataset_to_testsize
dataset_to_testsize["sinusoid"]="100 200 500 1000 2000"
dataset_to_testsize["sinusoid.2000"]="200"
dataset_to_testsize["sinusoid.4000"]="400"
dataset_to_testsize["sinusoid.6000"]="600"
dataset_to_testsize["sinusoid.8000"]="800"
dataset_to_testsize["sinusoid.10000"]="1000"

# method:models
declare -A method_to_model
method_to_model["hsic"]=$(printf "%s:mlp1x8x12x8-squared;" "${!dataset_to_testsize[@]}")
method_to_model["hsic-tied"]=$(printf "%s:mlp1x8x12x8-tied;" "${!dataset_to_testsize[@]}")
method_to_model["mmd"]=$(printf "%s:id-id@mlp2x8x12x8;" "${!dataset_to_testsize[@]}")
method_to_model["c2st"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["c2st-s"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["c2st-l"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["infonce"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["nwj"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["bandwidth"]=$(printf "%s:bandwidth-squared;" "${!dataset_to_testsize[@]}")


function train_args {
    case $method in
        bandwidth | hsic-tied)
            echo "\
                --train-config $train_root/hsic/train.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/hsic/$model/$run \
                --n-epochs 1000"
            ;;
        c2st-s | c2st-l)
            echo "\
                --train-config $train_root/c2st/train.c2st.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/c2st/$model/$run \
                --n-epochs 1000"
            ;;
        hsic-raw)
            echo "\
                --train-config $train_root/hsic/train.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/hsic_raw/$model/$run \
                --n-epochs 1000"
            ;;
        *)
            echo "\
                --train-config $train_root/$method/train.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/$method/$model/$run \
                --n-epochs 1000"
            ;;
    esac
}
function eval_args {
    case $method in
        bandwidth | hsic-tied)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/sinusoid/$dataset/hsic/$model/$run/best.pt \
                --log-dir $log_root/sinusoid/$run \
                --n-samples $n_samples"
            ;;
        c2st-s)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.acc.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/sinusoid/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/sinusoid/$run \
                --n-samples $n_samples"
            ;;
        c2st-l)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.logit.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/sinusoid/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/sinusoid/$run \
                --n-samples $n_samples"
            ;;
        hsic-raw)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/sinusoid/$dataset/hsic_raw/$model/$run/best.pt \
                --log-dir $log_root/sinusoid/$run \
                --n-samples $n_samples"
            ;;
        *)
            echo "\
                --eval-config $eval_root/$method/eval.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --pretrained-path $save_root/sinusoid/$dataset/$method/$model/$run/best.pt \
                --log-dir $log_root/sinusoid/$run \
                --n-samples $n_samples"
            ;;
    esac
}

run=1
datasets="sinusoid.10000"
source train.sh $run "bandwidth" "$datasets"
source eval.sh $run "bandwidth" "$datasets"



unset dataset_to_testsize
unset method_to_model
module purge
deactivate
