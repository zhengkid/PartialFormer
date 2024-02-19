#!/usr/bin/bash
set -e

model_root_dir=checkpoints

# set task
#task=iwslt-de2en
task=cnndm-32k
# set tag
model_dir_tag=pre_6_partialformer_A_G_MHSA_A_L_MHSA_HEAD_8_EXPAND_HEAD_16_16_DIM_360

# set device
gpu=(0 1 2 3 4 5 6 7)
cpu=

# data set
who=test
upper_bound=23
lower_bound=15
start_avg_num=5

if [ $task == "cnndm-32k" ]; then
        data_dir=cnn_10
        ensemble=5
        batch_size=64
        beam=4
        length_penalty=2.0

else
        echo "unknown task=$task"
        exit
fi

model_dir=$model_root_dir/$task/$model_dir_tag


if [ -n "$ensemble" ]; then
        if [ ! -d "$model_dir/ensemble_ckpts" ]; then
                mkdir -p $model_dir/ensemble_ckpts
                PYTHONPATH=`pwd` python3 scripts/enumerate_checkpoints.py --inputs $model_dir --output $model_dir/ensemble_ckpts --checkpoint-upper-bound $upper_bound --checkpoint-lower-bound $lower_bound --start-avg-num $start_avg_num
        fi
fi

if [ ! -d "$model_dir/hypos" ]; then
                mkdir -p $model_dir/hypos
fi
hypo_dir=$model_dir/hypos
ls $model_dir/ensemble_ckpts > $hypo_dir/ckpt_path_file
result=$hypo_dir/result.txt

len_gpu=${#gpu[@]}
len_ckpt=`cat $hypo_dir/ckpt_path_file | wc -l`
echo $len_gpu $len_ckpt
avg=$[$len_ckpt / $len_gpu]
if [ $(($len_ckpt % $len_gpu)) -ne 0 ]; then
	avg=$(($avg+1))
fi
split -l $avg $hypo_dir/ckpt_path_file -d -a 1 $hypo_dir/split_
for((i=0;i<$len_gpu;i++))
do {
if [ -n "$cpu" ]; then
    use_cpu=--cpu
fi

export CUDA_VISIBLE_DEVICES=${gpu[$i]}
for ckpt in `cat $hypo_dir/split_$i`
do
    ckpt_tag=`echo $ckpt | grep -o '[0-9]\+-[0-9]\+'`
    output=$hypo_dir/translation$ckpt_tag.log

    score=`cat $output | grep -o 'BLEU4 = [0-9]\+.[0-9]\+'`
    echo "$ckpt  $score" >> $result
    python3 rerank.py $hypo_dir/hypo$ckpt_tag.txt $hypo_dir/hypo$ckpt_tag.sorted

    test_name=cnndm.test.target

    # Tokenize.
    perl $tokenizer -l en < $hypo_dir/hypo$ckpt_tag.sorted > $hypo_dir/hypo$ckpt_tag.sorted.tok

    # Get rouge scores
    python3 get_rouge.py --decodes_filename $hypo_dir/hypo$ckpt_tag.sorted.tok --targets_filename cnndm.test.target.tok > $hypo_dir/rouge$ckpt_tag.log


!
done
}&
done
wait
echo "done decode"


