#!/bin/bash
#SBATCH --account=def-dsuth
#SBATCH --gpus-per-node=1               # Request 1 available GPU (--gpus-per-node=p100:1)
#SBATCH --mem=12000M                    # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-24:00:00               # DD-HH:MM:SS
#SBATCH --job-name=riab
#SBATCH --output=logs/%x/slurm-%j.out   # output file. %x is the job name, %N is the hostname, %j is the job id


# NOTE: this only works if each "sbatch" corresponds to a different bash shell.
# Otherwise the datasets variable will update for all "sbatch" instances.
# test if this script works by running a dataset exp and test size exp simutaneously

# config files
train_root=config/exp/train/
eval_root=config/exp/eval/
data_root=config/dataset/
model_root=config/model/
save_root=exp/pretrained/
log_root=exp/eval/


# dataset:testsizes
declare -A dataset_to_testsize
dataset_to_testsize["riab.present"]="100 200 500 1000 2000"
dataset_to_testsize["riab.present.500"]="200"
dataset_to_testsize["riab.present.1000"]="500"
dataset_to_testsize["riab.present.2000"]="1000"
dataset_to_testsize["riab.present.3000"]="1000"
dataset_to_testsize["riab.present.4000"]="1000"
dataset_to_testsize["riab.present.5000"]="2000"

# method:models
declare -A method_to_model
method_to_model["hsic"]=$(printf "%s:mlp8x32x64x32-mlp2x4x8x4;" "${!dataset_to_testsize[@]}")
method_to_model["mmd"]=$(printf "%s:id-id@mlp10x32x64x32;" "${!dataset_to_testsize[@]}")
method_to_model["c2st"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["c2st-s"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["c2st-l"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["infonce"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["bandwidth"]=$(printf "%s:bandwidth-squared;" "${!dataset_to_testsize[@]}")


function train_args {
    local basemethod
    case $method in
        bandwidth) basemethod="hsic" ;;
        c2st-s | c2st-l) basemethod="c2st" ;;
        *) basemethod=$method ;;
    esac
    echo "\
        --train-config $train_root/$basemethod/train.$basemethod.batch128.adamw.1e-4.yml \
        --data-config $data_root/riab/$dataset.yml \
        --model-config $model_root/$basemethod/$model.yml \
        --save-dir $save_root/riab/$dataset/$basemethod/$model/$run \
        --n-epochs 100 \
    "
}
function eval_args {
    local basemethod
    local eval_config
    case $method in
        bandwidth)
            basemethod="hsic"
            eval_config="$eval_root/$basemethod/eval.$basemethod.batch128.adamw.1e-4.yml"
            ;;
        c2st-s)
            basemethod="c2st"
            eval_config="$eval_root/c2st/eval.c2st.acc.batch128.adamw.1e-4.yml"
            ;;
        c2st-l)
            basemethod="c2st"
            eval_config="$eval_root/c2st/eval.c2st.logit.batch128.adamw.1e-4.yml"
            ;;
        *)
            basemethod=$method
            eval_config="$eval_root/$basemethod/eval.$basemethod.batch128.adamw.1e-4.yml"
            ;;
    esac
    echo "\
        --eval-config $eval_config \
        --data-config $data_root/riab/$dataset.yml \
        --model-config $model_root/$basemethod/$model.yml \
        --pretrained-path $save_root/riab/$dataset/$basemethod/$model/$run/best.pt \
        --log-dir $log_root/riab/$run \
        --n-samples $n_samples \
    "
}

run=1

# datasets="riab.present"
# datasets="riab.present.500 riab.present.1000 riab.present.2000 riab.present.3000 riab.present.4000 riab.present.5000"

# source train.sh $run "hsic bandwidth" "$datasets"
# source eval.sh $run "hsic bandwidth" "$datasets"

# source train.sh $run "c2st" "$datasets"
# source eval.sh $run "c2st-l c2st-s" "$datasets"

# source train.sh $run "mmd" "$datasets"
# source eval.sh $run "mmd" "$datasets"

# source train.sh $run "infonce" "$datasets"
# source eval.sh $run "infonce" "$datasets"




unset dataset_to_testsize
unset method_to_model
return
