#!/bin/bash
#SBATCH --account=def-dsuth
#SBATCH --gpus-per-node=1       # Request 1 available GPU (--gpus-per-node=p100:1)
#SBATCH --mem=4000M             # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-24:00:00       # DD-HH:MM:SS
#SBATCH --job-name=telco
#SBATCH --output=logs/%x/slurm-%j.out   # output file. %x is the job name, %N is the hostname, %j is the job id

# PROJ_DIR=$project/deepkernel
# VENV_DIR=$PROJ_DIR/myenv
# SOURCE_DIR=$PROJ_DIR/src
# SCRIPT_DIR=$PROJ_DIR/scripts
# cd $SCRIPT_DIR

# source $VENV_DIR/bin/activate
# module load python/3.10 scipy-stack/2023b cuda cudnn

# config files
train_root=config/exp/train/
eval_root=config/exp/eval/
data_root=config/dataset/
model_root=config/model/
save_root=exp/pretrained/
log_root=exp/eval/


# dataset:testsizes
declare -A dataset_to_testsize
dataset_to_testsize["telco"]="50 100 200 500 1000 2000"
dataset_to_testsize["telco.1000"]="400"
dataset_to_testsize["telco.2000"]="800"
dataset_to_testsize["telco.3000"]="1200"
dataset_to_testsize["telco.4000"]="1600"
dataset_to_testsize["telco.5000"]="2000"

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
method_to_model["mi"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["bandwidth"]=$(printf "%s:bandwidth-squared;" "${!dataset_to_testsize[@]}")


function train_args {
    case $method in
        bandwidth | hsic-tied)
            echo "\
                --train-config $train_root/hsic/train.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/telco/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/telco/$dataset/hsic/$model/$run \
                --n-epochs 2000"
            ;;
        c2st-s | c2st-l)
            echo "\
                --train-config $train_root/c2st/train.c2st.batch512.adamw.1e-4.yml \
                --data-config $data_root/telco/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --save-dir $save_root/telco/$dataset/c2st/$model/$run \
                --n-epochs 2000"
            ;;
        hsic-raw)
            echo "\
                --train-config $train_root/hsic/train.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/telco/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/telco/$dataset/hsic_raw/$model/$run \
                --n-epochs 2000"
            ;;
        *)
            echo "\
                --train-config $train_root/$method/train.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/telco/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --save-dir $save_root/telco/$dataset/$method/$model/$run \
                --n-epochs 5"
            ;;
    esac
}
function eval_args {
    case $method in
        bandwidth | hsic-tied)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/telco/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/telco/$dataset/hsic/$model/$run/best.pt \
                --log-dir $log_root/telco/$run \
                --n-samples $n_samples"
            ;;
        c2st-s)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.acc.batch512.adamw.1e-4.yml \
                --data-config $data_root/telco/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/telco/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/telco/$run \
                --n-samples $n_samples"
            ;;
        c2st-l)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.logit.batch512.adamw.1e-4.yml \
                --data-config $data_root/telco/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/telco/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/telco/$run \
                --n-samples $n_samples"
            ;;
        hsic-raw)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/telco/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/telco/$dataset/hsic_raw/$model/$run/best.pt \
                --log-dir $log_root/telco/$run \
                --n-samples $n_samples"
            ;;
        *)
            echo "\
                --eval-config $eval_root/$method/eval.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/telco/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --pretrained-path $save_root/telco/$dataset/$method/$model/$run/best.pt \
                --log-dir $log_root/telco/$run \
                --n-samples $n_samples"
            ;;
    esac
}

# run=power_vs_datasize/1
# datasets="telco.1000 telco.2000 telco.3000 telco.4000"
# for item in $datasets; do
#     source train.sh $run "mi" "$item"
#     source eval.sh $run "mi" "$item"
# done

run=power_vs_testsize/1
datasets="telco"
source train.sh $run "hsic" "$datasets"
source eval.sh $run "hsic" "$datasets"


unset dataset_to_testsize
unset method_to_model
# module purge
# deactivate
