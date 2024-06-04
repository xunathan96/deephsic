#!/bin/bash
# salloc --account=def-dsuth --gres=gpu:1 --cpus-per-task=3 --mem=32000M --time=1:00:00
# salloc --account=def-dsuth --gres=gpu:p100:1 --ntasks=8 --mem=16G --time=0:30:0 --nodes=1

# dictionary
declare -A shapes=(["circle"]=1 ["square"]=4)
shapes["triangle"]=3
shapes["pentagon"]=5
# shapes["pentagon"]=(23 33 1)    # bash does not support multi-dimensional arrays

echo "keys:     ${!shapes[@]}"    # expand keys
echo "values:   ${shapes[@]}"     # expand values


# hack for dictionary with list values
declare -A animals
animals["cat"]="4 2 1"
animals["dog"]="4 2 1"
animals["spider"]="9 0 0 12"

for animal in "${!animals[@]}"; do
    s=(${animals[$animal]})     # str > array
    echo "${s[@]}"
done


# hack for nested dictionary w/o array keys
declare -A animals
animals["cat"]="legs:4;ears:2;eyes:2"
animals["dog"]="legs:4;ears:2;eyes:2"
animals["spider"]="legs:9;ears:0;eyes:12"

for animal in "${!animals[@]}"; do
    dict=(${animals[$animal]//;/ })    # replace ';' with ' ' and convert to array
    echo $dict
done


# hack for nested dictionary w/ array keys
declare -A animals
animals["cat"]="names:popo bug;legs:4;ears:2;eyes:2"
animals["dog"]="names:oreo momo;legs:4;ears:2;eyes:2"
animals["spider"]="names:a b c d;legs:9;ears:0;eyes:12"


for animal in "${!animals[@]}"; do
    IFS=';'; dict=(${animals[$animal]}); unset IFS;    # use ';' as the internal field separator
    # echo $dict

    for item in "${dict[@]}"; do
        key="${item%%:*}"
        value="${item##*:}"
        echo "$key: $value"
    done

done



# string concat
arr=("A" "B" "C" "D")
printf -v a "<key:%s>\n<key:%s>\n<key:%s>\n<key:%s>\n" "${arr[@]}"
echo "$a"

arr=("A" "B" "C" "D")
printf -v b "<key:%s>\n" "${arr[@]}"
echo "$b"
