#!/bin/bash
#SBATCH --account=def-dsuth
#SBATCH --gpus-per-node=1       # Request 1 available GPU (--gpus-per-node=p100:1)
#SBATCH --mem=4000M             # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-24:00:00       # DD-HH:MM:SS
#SBATCH --job-name=sinusoid
#SBATCH --output=logs/%x/slurm-%j.out   # output file. %x is the job name, %N is the hostname, %j is the job id

PROJ_DIR=$project/deepkernel
VENV_DIR=$PROJ_DIR/myenv
SOURCE_DIR=$PROJ_DIR/src
SCRIPT_DIR=$PROJ_DIR/scripts
cd $SCRIPT_DIR

source $VENV_DIR/bin/activate
module load python/3.10 scipy-stack/2023b cuda cudnn

# config files
train_root=config/exp/train/
eval_root=config/exp/eval/
data_root=config/dataset/
model_root=config/model/
save_root=exp/pretrained/
log_root=exp/eval/


# dataset:testsizes
declare -A dataset_to_testsize
dataset_to_testsize["sinusoid"]="50 100 200 500 1000 2000"
dataset_to_testsize["sinusoid.1000"]="100"
dataset_to_testsize["sinusoid.2000"]="200"
dataset_to_testsize["sinusoid.3000"]="300"
dataset_to_testsize["sinusoid.4000"]="400"
dataset_to_testsize["sinusoid.6000"]="600"
dataset_to_testsize["sinusoid.8000"]="800"
dataset_to_testsize["sinusoid.10000"]="1000"

# method:models
declare -A method_to_model
method_to_model["hsic"]=$(printf "%s:mlp1x8x12x8-squared;" "${!dataset_to_testsize[@]}")
method_to_model["hsic-tied"]=$(printf "%s:mlp1x8x12x8-tied;" "${!dataset_to_testsize[@]}")
method_to_model["hsic-w/"]="${method_to_model["hsic"]}"
method_to_model["mmd"]=$(printf "%s:id-id@mlp2x8x12x8;" "${!dataset_to_testsize[@]}")
method_to_model["c2st"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["c2st-s"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["c2st-l"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["infonce"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["nwj"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["mi"]=$(printf "%s:id-id@mlp2x8x12x8x1;" "${!dataset_to_testsize[@]}")
method_to_model["nds"]="${method_to_model["infonce"]}"
method_to_model["nds-w/"]="${method_to_model["infonce"]}"
method_to_model["bandwidth"]=$(printf "%s:bandwidth-squared;" "${!dataset_to_testsize[@]}")


function train_args {
    case $method in
        bandwidth | hsic-tied)
            echo "\
                --train-config $train_root/hsic/train.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/hsic/$model/$run \
                --n-epochs 10000"
            ;;
        c2st-s | c2st-l)
            echo "\
                --train-config $train_root/c2st/train.c2st.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/c2st/$model/$run \
                --n-epochs 10000"
            ;;
        hsic-raw)
            echo "\
                --train-config $train_root/hsic/train.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/hsic_raw/$model/$run \
                --n-epochs 10000"
            ;;
        hsic-w/)
            echo "\
                --train-config $train_root/hsic/train.power_w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/hsic_w_thresh/$model/$run \
                --n-epochs 10000"
            ;;
        nds-w/)
            echo "\
                --train-config $train_root/nds/train.nds.w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/nds/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/nds_w_thresh/$model/$run \
                --n-epochs 10000"
            ;;
        *)
            echo "\
                --train-config $train_root/$method/train.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --save-dir $save_root/sinusoid/$dataset/$method/$model/$run \
                --n-epochs 4000"    # 10000 hsic/c2st/...; 4000 for mi (b/c mi too expensive).. actually 10000 seems okay? ~3h
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
        hsic-w/)
            echo "\
                --eval-config $eval_root/hsic/eval.power_w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/sinusoid/$dataset/hsic_w_thresh/$model/$run/best.pt \
                --log-dir $log_root/sinusoid/$run \
                --n-samples $n_samples"
            ;;
        nds-w/)
            echo "\
                --eval-config $eval_root/nds/eval.nds.w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/sinusoid/$dataset.yml \
                --model-config $model_root/nds/$model.yml \
                --pretrained-path $save_root/sinusoid/$dataset/nds_w_thresh/$model/$run/best.pt \
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

# run=1
# datasets="sinusoid.1000 sinusoid.2000 sinusoid.4000 sinusoid.3000"

# source train.sh $run "hsic hsic-tied bandwidth" "$datasets"
# source eval.sh $run "hsic hsic-tied bandwidth" "$datasets"
# dataset_to_testsize["sinusoid.4000"]="50 100 200 500 1000 2000"
# source eval.sh $run "hsic hsic-tied bandwidth" "$datasets"

# source train.sh $run "c2st" "$datasets"
# source eval.sh $run "c2st-l c2st-s" "$datasets"
# dataset_to_testsize["sinusoid.4000"]="50 100 200 500 1000 2000"
# source eval.sh $run "c2st-l c2st-s" "$datasets"

# source train.sh $run "mmd" "$datasets"
# source eval.sh $run "mmd" "$datasets"
# dataset_to_testsize["sinusoid.4000"]="50 100 200 500 1000 2000"
# source eval.sh $run "mmd" "$datasets"

# source train.sh $run "infonce" "$datasets"
# source eval.sh $run "infonce" "$datasets"
# dataset_to_testsize["sinusoid.4000"]="50 100 200 500 1000 2000"
# source eval.sh $run "infonce" "$datasets"

# source train.sh $run "nwj" "$datasets"
# source eval.sh $run "nwj" "$datasets"
# dataset_to_testsize["sinusoid.4000"]="50 100 200 500 1000 2000"
# source eval.sh $run "nwj" "$datasets"

# ----------- Mutual Information Tests -----------
# runs 7/8/9 are with minus trace (which fails)
# runs 10/11/12 are with minus trace/(n*n-1)
# runs 13/14/15 are with nce-like code

# run=power_vs_datasize/15
# datasets="sinusoid.1000 sinusoid.2000 sinusoid.3000 sinusoid.4000"
# for item in $datasets; do
#     source train.sh $run "mi" "$item"
#     source eval.sh $run "mi" "$item"
# done

# run=power_vs_testsize/15
# datasets="sinusoid"
# source train.sh $run "mi" "$datasets"
# source eval.sh $run "mi" "$datasets"


# ----------- HSIC w/ thresh -----------
# run=power_vs_testsize/3
# datasets="sinusoid"
# source train.sh $run "hsic-w/" "$datasets"
# source eval.sh $run "hsic-w/" "$datasets"

# run=power_vs_datasize/3
# datasets="sinusoid.1000 sinusoid.2000 sinusoid.3000 sinusoid.4000"
# for item in $datasets; do
#     source train.sh $run "hsic-w/" "$item"
#     source eval.sh $run "hsic-w/" "$item"
# done

# ----------- HSIC w/ thresh -----------
# run=power_vs_testsize/3
# datasets="sinusoid"
# source train.sh $run "nds-w/" "$datasets"
# source eval.sh $run "nds-w/" "$datasets"

# run=power_vs_datasize/1
# datasets="sinusoid.1000 sinusoid.2000 sinusoid.3000 sinusoid.4000"
# for item in $datasets; do
#     source train.sh $run "nds-w/" "$item"
#     source eval.sh $run "nds-w/" "$item"
# done


unset dataset_to_testsize
unset method_to_model
module purge
deactivate
