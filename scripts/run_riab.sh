#!/bin/bash
#SBATCH --account=def-dsuth
#SBATCH --gpus-per-node=1       # Request 1 available GPU (--gpus-per-node=p100:1)
#SBATCH --mem=4000M             # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-24:00:00       # DD-HH:MM:SS
#SBATCH --job-name=riab
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
dataset_to_testsize["riab.present"]="100 200 500 1000 2000"
dataset_to_testsize["riab.present.500"]="200"
dataset_to_testsize["riab.present.1000"]="400"
dataset_to_testsize["riab.present.2000"]="800"
dataset_to_testsize["riab.present.3000"]="1200"
dataset_to_testsize["riab.present.4000"]="1600"
dataset_to_testsize["riab.present.5000"]="2000"

# method:models
declare -A method_to_model
method_to_model["hsic"]=$(printf "%s:mlp8x32x64x32-mlp2x4x8x4;" "${!dataset_to_testsize[@]}")
method_to_model["hsic-raw"]="${method_to_model["hsic"]}"
method_to_model["hsic-w/"]="${method_to_model["hsic"]}"
method_to_model["mmd"]=$(printf "%s:id-id@mlp10x32x64x32;" "${!dataset_to_testsize[@]}")
method_to_model["c2st"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["c2st-s"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["c2st-l"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["infonce"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["nwj"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["mi"]=$(printf "%s:id-id@mlp10x32x64x32x1;" "${!dataset_to_testsize[@]}")
method_to_model["nds"]="${method_to_model["infonce"]}"
method_to_model["nds-w/"]="${method_to_model["infonce"]}"
method_to_model["bandwidth"]=$(printf "%s:bandwidth-squared;" "${!dataset_to_testsize[@]}")


function train_args {
    case $method in
        bandwidth)
            echo "\
                --train-config $train_root/hsic/train.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/riab/$dataset/hsic/$model/$run \
                --n-epochs 2000"
            ;;
        c2st-s | c2st-l)
            echo "\
                --train-config $train_root/c2st/train.c2st.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --save-dir $save_root/riab/$dataset/c2st/$model/$run \
                --n-epochs 2000"
            ;;
        hsic-raw)
            echo "\
                --train-config $train_root/hsic/train.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/riab/$dataset/hsic_raw/$model/$run \
                --n-epochs 2000"
            ;;
        hsic-w/)
            echo "\
                --train-config $train_root/hsic/train.power_w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/riab/$dataset/hsic_w_thresh/$model/$run \
                --n-epochs 2000"
            ;;
        nds-w/)
            echo "\
                --train-config $train_root/nds/train.nds.w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/nds/$model.yml \
                --save-dir $save_root/riab/$dataset/nds_w_thresh/$model/$run \
                --n-epochs 2000"
            ;;
        *)
            echo "\
                --train-config $train_root/$method/train.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --save-dir $save_root/riab/$dataset/$method/$model/$run \
                --n-epochs 2000"
            ;;
    esac
}
function eval_args {
    case $method in
        bandwidth)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/riab/$dataset/hsic/$model/$run/best.pt \
                --log-dir $log_root/riab/$run \
                --n-samples $n_samples"
            ;;
        c2st-s)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.acc.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/riab/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/riab/$run \
                --n-samples $n_samples"
            ;;
        c2st-l)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.logit.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/riab/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/riab/$run \
                --n-samples $n_samples"
            ;;
        hsic-raw)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/riab/$dataset/hsic_raw/$model/$run/best.pt \
                --log-dir $log_root/riab/$run \
                --n-samples $n_samples"
            ;;
        hsic-w/)
            echo "\
                --eval-config $eval_root/hsic/eval.power_w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/riab/$dataset/hsic_w_thresh/$model/$run/best.pt \
                --log-dir $log_root/riab/$run \
                --n-samples $n_samples"
            ;;
        nds-w/)
            echo "\
                --eval-config $eval_root/nds/eval.nds.w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/nds/$model.yml \
                --pretrained-path $save_root/riab/$dataset/nds_w_thresh/$model/$run/best.pt \
                --log-dir $log_root/riab/$run \
                --n-samples $n_samples"
            ;;
        *)
            echo "\
                --eval-config $eval_root/$method/eval.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/riab/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --pretrained-path $save_root/riab/$dataset/$method/$model/$run/best.pt \
                --log-dir $log_root/riab/$run \
                --n-samples $n_samples"
            ;;
    esac
}


# run=35
# runs 1,2,3 are for rate tests and 4,5,6 are size tests
#      7,8,9                        10,11,12
# runs 13,14,15 are both tests but with no validation for riab.present

# runs 16,17,18,19,20,21 are rate & size tests with new initialization
# runs 22,23,24,25,26,27 are size tests with consistent splits

# NWJ
# runs 28,29,30,31 are rate tests
# runs 32,33,34,35 are size tests

# datasets="riab.present"
# source train.sh $run "hsic c2st mmd infonce bandwidth hsic-raw" "$datasets"
# source eval.sh $run "hsic c2st-l c2st-s mmd infonce bandwidth hsic-raw" "$datasets"
# source train.sh $run "nwj" "$datasets"
# source eval.sh $run "nwj" "$datasets"

# datasets="riab.present.500 riab.present.1000 riab.present.2000 riab.present.3000 riab.present.4000 riab.present.5000"
# source train.sh $run "hsic c2st mmd infonce bandwidth hsic-raw" "$datasets"
# source eval.sh $run "hsic c2st-l c2st-s mmd infonce bandwidth hsic-raw" "$datasets"
# source train.sh $run "nwj" "$datasets"
# source eval.sh $run "nwj" "$datasets"


# ----------- Mutual Information Tests -----------
# runs 7/8/9 are with minus trace (which fails)
# runs 10/11/12 are with minus trace/(n*n-1)
# runs 13/14/15 are with nce-like code
# runs 16/17/18/19/20/21 are with batch=512 instead of 128

# run=power_vs_datasize/21
# datasets="riab.present.500 riab.present.1000 riab.present.2000 riab.present.3000 riab.present.4000 riab.present.5000"
# for item in $datasets; do
#     source train.sh $run "mi" "$item"
#     source eval.sh $run "mi" "$item"
# done

# ----------- HSIC w/ thresh -----------
# run=power_vs_testsize/3
# datasets="riab.present"
# source train.sh $run "hsic-w/" "$datasets"
# source eval.sh $run "hsic-w/" "$datasets"

# run=power_vs_datasize/3
# datasets="riab.present.500 riab.present.1000 riab.present.2000 riab.present.3000 riab.present.4000 riab.present.5000"
# for item in $datasets; do
#     source train.sh $run "hsic-w/" "$item"
#     source eval.sh $run "hsic-w/" "$item"
# done

# ----------- NDS w/ thresh -----------
# run=power_vs_testsize/3
# datasets="riab.present"
# source train.sh $run "nds-w/" "$datasets"
# source eval.sh $run "nds-w/" "$datasets"

# run=power_vs_datasize/3
# datasets="riab.present.500 riab.present.1000 riab.present.2000 riab.present.3000 riab.present.4000 riab.present.5000"
# for item in $datasets; do
#     source train.sh $run "nds-w/" "$item"
#     source eval.sh $run "nds-w/" "$item"
# done


unset dataset_to_testsize
unset method_to_model
module purge
deactivate
