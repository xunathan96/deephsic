#!/bin/bash
#SBATCH --account=def-dsuth
#SBATCH --gpus-per-node=1       # Request 1 available GPU (--gpus-per-node=p100:1)
#SBATCH --mem=4000M             # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-24:00:00       # DD-HH:MM:SS
#SBATCH --job-name=wine
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
dataset_to_testsize["wine"]="50 100 200 500 1000 2000"
dataset_to_testsize["wine.500"]="200"
dataset_to_testsize["wine.1000"]="400"
dataset_to_testsize["wine.1200"]="480"
dataset_to_testsize["wine.1300"]="520"
dataset_to_testsize["wine.1500"]="600"
dataset_to_testsize["wine.2000"]="800"

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
method_to_model["nds-w/"]="${method_to_model["infonce"]}"
method_to_model["bandwidth"]=$(printf "%s:bandwidth-squared;" "${!dataset_to_testsize[@]}")


function train_args {
    case $method in
        bandwidth | hsic-tied)
            echo "\
                --train-config $train_root/hsic/train.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/wine/$dataset/hsic/$model/$run \
                --n-epochs 10000"
            ;;
        c2st-s | c2st-l)
            echo "\
                --train-config $train_root/c2st/train.c2st.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --save-dir $save_root/wine/$dataset/c2st/$model/$run \
                --n-epochs 10000"
            ;;
        nds-w/)
            echo "\
                --train-config $train_root/nds/train.nds.w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/nds/$model.yml \
                --save-dir $save_root/wine/$dataset/nds_w_thresh/$model/$run \
                --n-epochs 10000"
            ;;
        *)
            echo "\
                --train-config $train_root/$method/train.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --save-dir $save_root/wine/$dataset/$method/$model/$run \
                --n-epochs 10000"
            ;;
    esac
}
function eval_args {
    case $method in
        bandwidth | hsic-tied)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/wine/$dataset/hsic/$model/$run/best.pt \
                --log-dir $log_root/wine/$run \
                --n-samples $n_samples"
            ;;
        c2st-s)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.acc.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/wine/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/wine/$run \
                --n-samples $n_samples"
            ;;
        c2st-l)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.logit.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/wine/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/wine/$run \
                --n-samples $n_samples"
            ;;
        hsic-raw)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/wine/$dataset/hsic_raw/$model/$run/best.pt \
                --log-dir $log_root/wine/$run \
                --n-samples $n_samples"
            ;;
        nds-w/)
            echo "\
                --eval-config $eval_root/nds/eval.nds.w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/nds/$model.yml \
                --pretrained-path $save_root/wine/$dataset/nds_w_thresh/$model/$run/best.pt \
                --log-dir $log_root/wine/$run \
                --n-samples $n_samples"
            ;;
        *)
            echo "\
                --eval-config $eval_root/$method/eval.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/wine/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --pretrained-path $save_root/wine/$dataset/$method/$model/$run/best.pt \
                --log-dir $log_root/wine/$run \
                --n-samples $n_samples"
            ;;
    esac
}

# # POWER VS DATASIZE

# run=power_vs_datasize/3
# datasets="wine.1200 wine.1300"
# for item in $datasets; do
#     source train.sh $run "hsic" "$item"
#     source eval.sh $run "hsic" "$item"
# done
# for item in $datasets; do
#     source train.sh $run "bandwidth" "$item"
#     source eval.sh $run "bandwidth" "$item"
# done

# run=power_vs_datasize/3
# datasets="wine.1200 wine.1300"
# for item in $datasets; do
#     source train.sh $run "mmd" "$item"
#     source eval.sh $run "mmd" "$item"
# done

# run=power_vs_datasize/3
# datasets="wine.1200 wine.1300"
# for item in $datasets; do
#     source train.sh $run "c2st" "$item"
#     source eval.sh $run "c2st-s c2st-l" "$item"
# done

# run=power_vs_datasize/3
# datasets="wine.1200 wine.1300"
# for item in $datasets; do
#     source train.sh $run "infonce" "$item"
#     source eval.sh $run "infonce" "$item"
# done

# run=power_vs_datasize/3
# datasets="wine.1200 wine.1300"
# for item in $datasets; do
#     source train.sh $run "nwj" "$item"
#     source eval.sh $run "nwj" "$item"
# done

# run=power_vs_datasize/3
# datasets="wine.1200 wine.1300"
# for item in $datasets; do
#     source train.sh $run "mi" "$item"
#     source eval.sh $run "mi" "$item"
# done


# # POWER VS TESTSIZE

# dataset_to_testsize["wine.2000"]="50 100 200 500 1000"
# run=power_vs_testsize/3
# datasets="wine.2000"
# source eval.sh $run "bandwidth mi" "$datasets"

# ----------- NDS w/ thresh -----------
# run=power_vs_testsize/1
# datasets="wine.2000"
# dataset_to_testsize["wine.2000"]="50 100 200 500 1000"
# for item in $datasets; do
#     source train.sh $run "nds-w/" "$item"
#     source eval.sh $run "nds-w/" "$item"
# done

# run=power_vs_datasize/1
# datasets="wine.500 wine.1000 wine.1200 wine.1300 wine.1500 wine.2000"
# for item in $datasets; do
#     source train.sh $run "nds-w/" "$item"
#     source eval.sh $run "nds-w/" "$item"
# done


unset dataset_to_testsize
unset method_to_model
module purge
deactivate
