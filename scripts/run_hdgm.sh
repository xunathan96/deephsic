#!/bin/bash
#SBATCH --account=def-dsuth
#SBATCH --gpus-per-node=1               # Request 1 available GPU (--gpus-per-node=p100:1)
#SBATCH --mem=12000M                    # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-24:00:00               # DD-HH:MM:SS
#SBATCH --job-name=hdgm
#SBATCH --output=logs/%x/slurm-%j.out   # output file. %x is the job name, %N is the hostname, %j is the job id

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
dataset_to_testsize["hdgm4.n1000"]="500"
dataset_to_testsize["hdgm4.n2000"]="1000"
dataset_to_testsize["hdgm4.n3000"]="1000"
dataset_to_testsize["hdgm4.n4000"]="1000"
dataset_to_testsize["hdgm8"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm8.n1000"]="500"
dataset_to_testsize["hdgm8.n2000"]="1000"
dataset_to_testsize["hdgm8.n3000"]="1000"
dataset_to_testsize["hdgm8.n4000"]="1000"
dataset_to_testsize["hdgm10"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm10.n2000"]="1000"
dataset_to_testsize["hdgm10.n4000"]="1000"
dataset_to_testsize["hdgm10.n6000"]="2000"
dataset_to_testsize["hdgm10.n8000"]="2000"
dataset_to_testsize["hdgm20"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm20.n4000"]="1000"
dataset_to_testsize["hdgm20.n8000"]="2000"
dataset_to_testsize["hdgm20.n12000"]="2000"
dataset_to_testsize["hdgm20.n16000"]="2000"
dataset_to_testsize["hdgm30"]="100 200 500 1000 2000"
dataset_to_testsize["hdgm30.n8000"]="2000"
dataset_to_testsize["hdgm30.n12000"]="2000"
dataset_to_testsize["hdgm30.n16000"]="2000"
dataset_to_testsize["hdgm30.n20000"]="2000"
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
method_to_model["bandwidth"]="$(printf "%s:bandwidth-squared;" "hdgm4" "hdgm4.n1000" "hdgm4.n2000" "hdgm4.n3000" "hdgm4.n4000")\
                              $(printf "%s:bandwidth-squared;" "hdgm8" "hdgm8.n1000" "hdgm8.n2000" "hdgm8.n3000" "hdgm8.n4000")\
                              $(printf "%s:bandwidth-squared;" "hdgm10" "hdgm10.n2000" "hdgm10.n4000" "hdgm10.n6000" "hdgm10.n8000")\
                              $(printf "%s:bandwidth-squared;" "hdgm20" "hdgm20.n4000" "hdgm20.n8000" "hdgm20.n12000" "hdgm20.n16000")\
                              $(printf "%s:bandwidth-squared;" "hdgm30" "hdgm30.n8000" "hdgm30.n12000" "hdgm30.n16000" "hdgm30.n20000")\
                              $(printf "%s:bandwidth-squared;" "hdgm40")\
                              $(printf "%s:bandwidth-squared;" "hdgm50")\
                              "


function train_args {
    local basemethod
    case $method in
        bandwidth) basemethod="hsic" ;;
        hsic-tied) basemethod="hsic" ;;
        c2st-s | c2st-l) basemethod="c2st" ;;
        *) basemethod=$method ;;
    esac
    echo "\
        --train-config $train_root/$basemethod/train.$basemethod.batch128.adamw.1e-4.yml \
        --data-config $data_root/hdgm/$dataset.yml \
        --model-config $model_root/$basemethod/$model.yml \
        --save-dir $save_root/hdgm/$dataset/$basemethod/$model/$run \
        --n-epochs 10 \
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
        hsic-tied)
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
        --data-config $data_root/hdgm/$dataset.yml \
        --model-config $model_root/$basemethod/$model.yml \
        --pretrained-path $save_root/hdgm/$dataset/$basemethod/$model/$run/best.pt \
        --log-dir $log_root/hdgm/$run \
        --n-samples $n_samples \
    "
}

run=1

# datasets="hdgm4 hdgm8 hdgm10 hdgm20 hdgm30 hdgm40 hdgm50"
# datasets="hdgm4.n1000 hdgm4.n2000 hdgm4.n3000 hdgm4.n4000 \
#           hdgm8.n1000 hdgm8.n2000 hdgm8.n3000 hdgm8.n4000 \
#           hdgm10.n2000 hdgm10.n4000 hdgm10.n6000 hdgm10.n8000 \
#           hdgm20.n4000 hdgm20.n8000 hdgm20.n12000 hdgm20.n16000 \
#           hdgm30.n8000 hdgm30.n12000 hdgm30.n16000 hdgm30.n20000 \
#           "

# source train.sh $run "hsic hsic-tied" "$datasets"
# source eval.sh $run "hsic hsic-tied" "$datasets"

# source train.sh $run "c2st" "$datasets"
# source eval.sh $run "c2st-s c2st-l" "$datasets"

# source train.sh $run "mmd" "$datasets"
# source eval.sh $run "mmd" "$datasets"

# source train.sh $run "infonce" "$datasets"
# source eval.sh $run "infonce" "$datasets"

# source train.sh $run "bandwidth" "$datasets"
# source eval.sh $run "bandwidth" "$datasets"



unset dataset_to_testsize
unset method_to_model
return
