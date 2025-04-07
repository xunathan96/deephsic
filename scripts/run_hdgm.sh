#!/bin/bash
#SBATCH --account=def-dsuth
#SBATCH --gpus-per-node=1       # Request 1 available GPU (--gpus-per-node=p100:1)
#SBATCH --mem=4000M             # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-48:00:00       # DD-HH:MM:SS
#SBATCH --job-name=hdgm
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
dataset_to_testsize["hdgm4"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm4.n1000"]="300"
dataset_to_testsize["hdgm4.n2000"]="600"
dataset_to_testsize["hdgm4.n3000"]="900"
dataset_to_testsize["hdgm4.n4000"]="1200"
dataset_to_testsize["hdgm8"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm8.n1000"]="300"
dataset_to_testsize["hdgm8.n2000"]="600"
dataset_to_testsize["hdgm8.n3000"]="900"
dataset_to_testsize["hdgm8.n4000"]="1200"
dataset_to_testsize["hdgm10"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm10.n2000"]="200" #"600"
dataset_to_testsize["hdgm10.n4000"]="400" #"1200"
dataset_to_testsize["hdgm10.n6000"]="600" #"1800"
dataset_to_testsize["hdgm10.n8000"]="800" #"2400"
dataset_to_testsize["hdgm20"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm20.n4000"]="400" #"1200"
dataset_to_testsize["hdgm20.n8000"]="800" #"2400"
dataset_to_testsize["hdgm20.n12000"]="1200" #"3600"
dataset_to_testsize["hdgm20.n16000"]="1600" #"4800"
dataset_to_testsize["hdgm30"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm30.n8000"]="1600" #"2400"
dataset_to_testsize["hdgm30.n12000"]="2400" #"3600"
dataset_to_testsize["hdgm30.n16000"]="3200" #"4800"
dataset_to_testsize["hdgm30.n20000"]="4000" #"6000"
dataset_to_testsize["hdgm40"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm50"]="100 200 500 1000 2000"


# method:models
declare -A method_to_model
method_to_model["hsic"]="$(printf "%s:mlp2x4x6x4-squared;" "hdgm4" "hdgm4.n1000" "hdgm4.n2000" "hdgm4.n3000" "hdgm4.n4000")\
                         $(printf "%s:mlp4x8x12x8-squared;" "hdgm8" "hdgm8.n1000" "hdgm8.n2000" "hdgm8.n3000" "hdgm8.n4000")\
                         $(printf "%s:mlp5x10x15x10-squared;" "hdgm10" "hdgm10.n2000" "hdgm10.n4000" "hdgm10.n6000" "hdgm10.n8000")\
                         $(printf "%s:mlp10x20x30x20-squared;" "hdgm20" "hdgm20.n4000" "hdgm20.n8000" "hdgm20.n12000" "hdgm20.n16000")\
                         $(printf "%s:mlp15x30x45x30-squared;" "hdgm30" "hdgm30.n8000" "hdgm30.n12000" "hdgm30.n16000" "hdgm30.n20000")\
                         $(printf "%s:mlp20x40x60x40-squared;" "hdgm40")\
                         $(printf "%s:mlp25x50x75x50-squared;" "hdgm50")\
                         "
method_to_model["hsic-tied"]="$(printf "%s:mlp2x4x6x4-tied;" "hdgm4" "hdgm4.n1000" "hdgm4.n2000" "hdgm4.n3000" "hdgm4.n4000")\
                              $(printf "%s:mlp4x8x12x8-tied;" "hdgm8" "hdgm8.n1000" "hdgm8.n2000" "hdgm8.n3000" "hdgm8.n4000")\
                              $(printf "%s:mlp5x10x15x10-tied;" "hdgm10" "hdgm10.n2000" "hdgm10.n4000" "hdgm10.n6000" "hdgm10.n8000")\
                              $(printf "%s:mlp10x20x30x20-tied;" "hdgm20" "hdgm20.n4000" "hdgm20.n8000" "hdgm20.n12000" "hdgm20.n16000")\
                              $(printf "%s:mlp15x30x45x30-tied;" "hdgm30" "hdgm30.n8000" "hdgm30.n12000" "hdgm30.n16000" "hdgm30.n20000")\
                              $(printf "%s:mlp20x40x60x40-tied;" "hdgm40")\
                              $(printf "%s:mlp25x50x75x50-tied;" "hdgm50")\
                              "
method_to_model["hsic-raw"]="${method_to_model["hsic"]}"
method_to_model["hsic-w/"]="${method_to_model["hsic"]}"
method_to_model["mmd"]="$(printf "%s:id-id@mlp4x8x12x8;" "hdgm4" "hdgm4.n1000" "hdgm4.n2000" "hdgm4.n3000" "hdgm4.n4000")\
                        $(printf "%s:id-id@mlp8x16x24x16;" "hdgm8" "hdgm8.n1000" "hdgm8.n2000" "hdgm8.n3000" "hdgm8.n4000")\
                        $(printf "%s:id-id@mlp10x20x30x20;" "hdgm10" "hdgm10.n2000" "hdgm10.n4000" "hdgm10.n6000" "hdgm10.n8000")\
                        $(printf "%s:id-id@mlp20x40x60x40;" "hdgm20" "hdgm20.n4000" "hdgm20.n8000" "hdgm20.n12000" "hdgm20.n16000")\
                        $(printf "%s:id-id@mlp30x60x90x60;" "hdgm30" "hdgm30.n8000" "hdgm30.n12000" "hdgm30.n16000" "hdgm30.n20000")\
                        $(printf "%s:id-id@mlp40x80x120x80;" "hdgm40")\
                        $(printf "%s:id-id@mlp50x100x150x100;" "hdgm50")\
                        "
method_to_model["c2st"]="$(printf "%s:id-id@mlp4x8x12x8x1;" "hdgm4" "hdgm4.n1000" "hdgm4.n2000" "hdgm4.n3000" "hdgm4.n4000")\
                         $(printf "%s:id-id@mlp8x16x24x16x1;" "hdgm8" "hdgm8.n1000" "hdgm8.n2000" "hdgm8.n3000" "hdgm8.n4000")\
                         $(printf "%s:id-id@mlp10x20x30x20x1;" "hdgm10" "hdgm10.n2000" "hdgm10.n4000" "hdgm10.n6000" "hdgm10.n8000")\
                         $(printf "%s:id-id@mlp20x40x60x40x1;" "hdgm20" "hdgm20.n4000" "hdgm20.n8000" "hdgm20.n12000" "hdgm20.n16000")\
                         $(printf "%s:id-id@mlp30x60x90x60x1;" "hdgm30" "hdgm30.n8000" "hdgm30.n12000" "hdgm30.n16000" "hdgm30.n20000")\
                         $(printf "%s:id-id@mlp40x80x120x80x1;" "hdgm40")\
                         $(printf "%s:id-id@mlp50x100x150x100x1;" "hdgm50")\
                         "
method_to_model["c2st-s"]="${method_to_model["c2st"]}"
method_to_model["c2st-l"]="${method_to_model["c2st"]}"
method_to_model["infonce"]="$(printf "%s:id-id@mlp4x8x12x8x1;" "hdgm4" "hdgm4.n1000" "hdgm4.n2000" "hdgm4.n3000" "hdgm4.n4000")\
                            $(printf "%s:id-id@mlp8x16x24x16x1;" "hdgm8" "hdgm8.n1000" "hdgm8.n2000" "hdgm8.n3000" "hdgm8.n4000")\
                            $(printf "%s:id-id@mlp10x20x30x20x1;" "hdgm10" "hdgm10.n2000" "hdgm10.n4000" "hdgm10.n6000" "hdgm10.n8000")\
                            $(printf "%s:id-id@mlp20x40x60x40x1;" "hdgm20" "hdgm20.n4000" "hdgm20.n8000" "hdgm20.n12000" "hdgm20.n16000")\
                            $(printf "%s:id-id@mlp30x60x90x60x1;" "hdgm30" "hdgm30.n8000" "hdgm30.n12000" "hdgm30.n16000" "hdgm30.n20000")\
                            $(printf "%s:id-id@mlp40x80x120x80x1;" "hdgm40")\
                            $(printf "%s:id-id@mlp50x100x150x100x1;" "hdgm50")\
                            "
method_to_model["nwj"]="${method_to_model["infonce"]}"
method_to_model["mi"]="${method_to_model["infonce"]}"
method_to_model["nds"]="${method_to_model["infonce"]}"
method_to_model["nds-w/"]="${method_to_model["infonce"]}"
method_to_model["bandwidth"]="$(printf "%s:bandwidth-squared;" "hdgm4" "hdgm4.n1000" "hdgm4.n2000" "hdgm4.n3000" "hdgm4.n4000")\
                              $(printf "%s:bandwidth-squared;" "hdgm8" "hdgm8.n1000" "hdgm8.n2000" "hdgm8.n3000" "hdgm8.n4000")\
                              $(printf "%s:bandwidth-squared;" "hdgm10" "hdgm10.n2000" "hdgm10.n4000" "hdgm10.n6000" "hdgm10.n8000")\
                              $(printf "%s:bandwidth-squared;" "hdgm20" "hdgm20.n4000" "hdgm20.n8000" "hdgm20.n12000" "hdgm20.n16000")\
                              $(printf "%s:bandwidth-squared;" "hdgm30" "hdgm30.n8000" "hdgm30.n12000" "hdgm30.n16000" "hdgm30.n20000")\
                              $(printf "%s:bandwidth-squared;" "hdgm40")\
                              $(printf "%s:bandwidth-squared;" "hdgm50")\
                              "

function train_args {
    case $method in
        bandwidth | hsic-tied)
            echo "\
                --train-config $train_root/hsic/train.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/hdgm/$dataset/hsic/$model/$run \
                --n-epochs 1000"
            ;;
        c2st-s | c2st-l)
            echo "\
                --train-config $train_root/c2st/train.c2st.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --save-dir $save_root/hdgm/$dataset/c2st/$model/$run \
                --n-epochs 1000"
            ;;
        hsic-raw)
            echo "\
                --train-config $train_root/hsic/train.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/hdgm/$dataset/hsic_raw/$model/$run \
                --n-epochs 1000"
            ;;
        hsic-w/)
            echo "\
                --train-config $train_root/hsic/train.power_w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --save-dir $save_root/hdgm/$dataset/hsic_w_thresh/$model/$run \
                --n-epochs 1000"
            ;;
        nds-w/)
            echo "\
                --train-config $train_root/nds/train.nds.w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/nds/$model.yml \
                --save-dir $save_root/hdgm/$dataset/nds_w_thresh/$model/$run \
                --n-epochs 2000"
            ;;
        *)
            echo "\
                --train-config $train_root/$method/train.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --save-dir $save_root/hdgm/$dataset/$method/$model/$run \
                --n-epochs 1000"    # 1000 for hsic/c2st/...; 2000 for mi (b/c results too poor)
            ;;
    esac
}
function eval_args {
    case $method in
        bandwidth | hsic-tied)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/hdgm/$dataset/hsic/$model/$run/best.pt \
                --log-dir $log_root/hdgm/$run \
                --n-samples $n_samples"
            ;;
        c2st-s)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.acc.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/hdgm/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/hdgm/$run \
                --n-samples $n_samples"
            ;;
        c2st-l)
            echo "\
                --eval-config $eval_root/c2st/eval.c2st.logit.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/c2st/$model.yml \
                --pretrained-path $save_root/hdgm/$dataset/c2st/$model/$run/best.pt \
                --log-dir $log_root/hdgm/$run \
                --n-samples $n_samples"
            ;;
        hsic-raw)
            echo "\
                --eval-config $eval_root/hsic/eval.hsic_raw.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/hdgm/$dataset/hsic_raw/$model/$run/best.pt \
                --log-dir $log_root/hdgm/$run \
                --n-samples $n_samples"
            ;;
        hsic-w/)
            echo "\
                --eval-config $eval_root/hsic/eval.power_w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/hsic/$model.yml \
                --pretrained-path $save_root/hdgm/$dataset/hsic_w_thresh/$model/$run/best.pt \
                --log-dir $log_root/hdgm/$run \
                --n-samples $n_samples"
            ;;
        nds-w/)
            echo "\
                --eval-config $eval_root/nds/eval.nds.w_thresh.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/nds/$model.yml \
                --pretrained-path $save_root/hdgm/$dataset/nds_w_thresh/$model/$run/best.pt \
                --log-dir $log_root/hdgm/$run \
                --n-samples $n_samples"
            ;;
        *)
            echo "\
                --eval-config $eval_root/$method/eval.$method.batch512.adamw.1e-4.yml \
                --data-config $data_root/hdgm/$dataset.yml \
                --model-config $model_root/$method/$model.yml \
                --pretrained-path $save_root/hdgm/$dataset/$method/$model/$run/best.pt \
                --log-dir $log_root/hdgm/$run \
                --n-samples $n_samples"
            ;;
    esac
}

# run=55
# runs 1,2,3 are for rate tests and 4,5,6 are size tests
# runs 7,8,9 are with new initializations (and 1000 epochs)
# runs 10,11,12 are with kaiming init

# runs 13,14,15 are rate tests and 16,17,18 are size tests (best inits: default for hsic, narrow_normal for others)
# runs 19,20,21 are rate tests and 22,23,24 are size tests (best inits, no validation, consistent 7:3 splits)
# runs 25,26,27 are rate tests and 28,29,30 are size tests (best inits, 2000 validation; 9:1 splits) 
# runs 31,32,33 are rate tests and 34,35,36 are size tests (best inits, 2000 validation; 7:1:2 splits)
# runs 37,38,39 are rate tests and 40,41,42 are size tests (best inits, 2000 validation; 7:2:1 splits)
# runs 43,44,45 are size tests (7:2:1 splits)

# adding nwj
# runs 25,26,27,31 are rate tests and 40,41,42,43 are size tests with best init (truc. normal)
# runs 37,38,39 are rate tests with default init

# testing higher batch size
# run hsic_narrow_1,2,3 are rate tests for hdgm50 (narrow_normal, 2000 validation, 512 batch size)
# runs 46,47,48,49,50 are rate tests (best inits, 2000 validation, 512 batch size)

# testing default init and 512 batch size
# runs 51,52,53,54,55 are rate tests (default init, 2000 validation, 512 batch size)


# datasets="hdgm4 hdgm8 hdgm10 hdgm20 hdgm30 hdgm40 hdgm50"
# datasets="hdgm4.n1000 hdgm4.n2000 hdgm4.n3000 hdgm4.n4000 \
#           hdgm8.n1000 hdgm8.n2000 hdgm8.n3000 hdgm8.n4000 \
#           hdgm10.n2000 hdgm10.n4000 hdgm10.n6000 hdgm10.n8000 \
#           hdgm20.n4000 hdgm20.n8000 hdgm20.n12000 hdgm20.n16000 \
#           hdgm30.n8000 hdgm30.n12000 hdgm30.n16000 hdgm30.n20000 \
#           "
# datasets="hdgm10.n2000 hdgm10.n4000 hdgm10.n6000 hdgm10.n8000 \
#           hdgm20.n4000 hdgm20.n8000 hdgm20.n12000 hdgm20.n16000 \
#           "

# source train.sh $run "hsic hsic-tied" "$datasets"
# source eval.sh $run "hsic hsic-tied" "$datasets"

# source train.sh $run "c2st" "$datasets"
# source eval.sh $run "c2st-s c2st-l" "$datasets"

# source train.sh $run "mmd" "$datasets"
# source eval.sh $run "mmd" "$datasets"

# source train.sh $run "infonce" "$datasets"
# source eval.sh $run "infonce" "$datasets"

# source train.sh $run "nwj" "$datasets"
# source eval.sh $run "nwj" "$datasets"

# source train.sh $run "hsic-raw bandwidth" "$datasets"
# source eval.sh $run "hsic-raw bandwidth" "$datasets"


# ----------- Mutual Information Tests -----------
# hdgm<20 takes 4h/1000 epochs
# hdgm>=20 takes 38h/1000 epochs
# hdgm 40/50 should use 500 epochs

# runs 4/5/6 new

# run=power_vs_datasize/6
# datasets="hdgm10.n2000 hdgm10.n4000 hdgm10.n6000 hdgm10.n8000"
# for item in $datasets; do
#     source train.sh $run "mi" "$item"
#     source eval.sh $run "mi" "$item"
# done

# run=power_vs_datasize/6
# datasets="hdgm20.n16000 hdgm20.n12000 hdgm20.n8000 hdgm20.n4000"
# for item in $datasets; do
#     source train.sh $run "mi" "$item"
#     source eval.sh $run "mi" "$item"
# done

# hdgm 4/8/10 are with 3000 epochs
# hdgm 20/30 are with 1000 epochs
# run=power_vs_testsize/6
# datasets="hdgm4 hdgm8 hdgm10 hdgm20 hdgm30 hdgm40 hdgm50"
# source train.sh $run "mi" "$datasets"
# source eval.sh $run "mi" "$datasets"


# ----------- InfoNCE -----------
# run=power_vs_testsize/3
# datasets="hdgm4 hdgm8 hdgm10 hdgm20 hdgm30 hdgm40 hdgm50"
# for item in $datasets; do
#     source train.sh $run "infonce" "$item"
#     source eval.sh $run "infonce" "$item"
# done

# run=power_vs_datasize/3
# datasets="hdgm10.n2000 hdgm10.n4000 hdgm10.n6000 hdgm10.n8000"
# for item in $datasets; do
#     source train.sh $run "infonce" "$item"
#     source eval.sh $run "infonce" "$item"
# done

# ----------- HSIC w/ thresh -----------
# run=power_vs_testsize/3
# datasets="hdgm4 hdgm8 hdgm10 hdgm20 hdgm30 hdgm40 hdgm50"
# for item in $datasets; do
#     source train.sh $run "hsic-w/" "$item"
#     source eval.sh $run "hsic-w/" "$item"
# done

# run=power_vs_datasize/3
# datasets="hdgm10.n2000 hdgm10.n4000 hdgm10.n6000 hdgm10.n8000"
# for item in $datasets; do
#     source train.sh $run "hsic-w/" "$item"
#     source eval.sh $run "hsic-w/" "$item"
# done

# ----------- NDS w/ thresh -----------
# run=power_vs_testsize/3
# datasets="hdgm4 hdgm8 hdgm10 hdgm20 hdgm30 hdgm40 hdgm50"
# for item in $datasets; do
#     source train.sh $run "nds-w/" "$item"
#     source eval.sh $run "nds-w/" "$item"
# done

# run=power_vs_datasize/3
# datasets="hdgm10.n2000 hdgm10.n4000 hdgm10.n6000 hdgm10.n8000"
# for item in $datasets; do
#     source train.sh $run "nds-w/" "$item"
#     source eval.sh $run "nds-w/" "$item"
# done



unset dataset_to_testsize
unset method_to_model
module purge
deactivate
