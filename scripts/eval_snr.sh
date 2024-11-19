#!/bin/bash
# cd $project/deepkernel/src
cd ../src

# wrap code with () to make all vars local
(
run=$1; shift
methods=($1); shift
datasets=($1); shift
if [[ "${methods[@]}" == "" ]]; then
    methods="${!method_to_model[@]}"
fi
if [[ "${datasets[@]}" == "" ]]; then
    datasets="${!dataset_to_testsize[@]}"
fi
# echo "evaluating <${methods[@]}> on <${datasets[@]}>..."


function get {
    # returns the value for a given key of a given dictionary
    # arguments:
    #   $1: dictionary as a string "key1:value1;key2:value2;..." with items separated by ';'
    #   $2: string key
    # return:
    #   string value for the given key
    IFS=';'; local dict=($1); unset IFS;    # split items by ';'
    local query=$2
    for item in "${dict[@]}"; do
        local raw_key="${item%%:*}"
        local raw_value="${item##*:}"
        local key=$(echo "$raw_key" | xargs)    # trim whitespace
        local value=$(echo "$raw_value" | xargs)
        if [ "$query" == "$key" ]; then
            echo "$value"
        fi
    done
}


for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        models=($(get "${method_to_model[$method]}" "$dataset"))
        sizes=(${dataset_to_testsize[$dataset]})
        for model in "${models[@]}"; do
            for n_samples in "${sizes[@]}"; do

                echo "evaluating $method: <$model> on <$dataset> with <$n_samples> test samples..."
                python eval_snr.py $(eval_args)

            done
        done
    done
done
)

cd -
