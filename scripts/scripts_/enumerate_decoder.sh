#!/usr/bin/bash
set -e

model_root_dir=checkpoints

# set task
#task=iwslt-de2en
task=cnndm-10k
# set tag
model_dir_tag=pre-6-gsan

# set device
gpu=(0 1 2 3 4 5 6 7)
cpu=

# data set
who=test
upper_bound=30
lower_bound=16
start_avg_num=5

if [ $task == "cnndm-10k" ]; then
        data_dir=cnndm-10k
        ensemble=$1
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

#export CUDA_VISIBLE_DEVICES=${gpu[$i]}
for ckpt in `cat $hypo_dir/split_$i`
do
    ckpt_tag=`echo $ckpt | grep -o '[0-9]\+-[0-9]\+'`
    output=$hypo_dir/translation$ckpt_tag.log

    for ((j=0;j<$len_gpu;j++))
    do
    {
        export CUDA_VISIBLE_DEVICES=$j
        python3 fairseq_cli/generate.py \
        data-bin/$data_dir \
        --path $model_dir/ensemble_ckpts/$ckpt \
        --gen-subset test$j \
        --fp16 \
        --task cnndm_bpe_phrase_translation \
        --truncate-source \
        --batch-size $batch_size \
        --beam $beam \
        --lenpen $length_penalty \
        --min-len 55 \
        --max-len-b 140 \
        --max-source-positions 500 \
        --no-repeat-ngram-size 3 \
        --output $hypo_dir/hypo${ckpt_tag}_$j.txt \
        --quiet \
        --remove-bpe | tee $output

        python3 rerank.py $hypo_dir/hypo${ckpt_tag}_$j.txt $hypo_dir/hypo${ckpt_tag}_$j.sorted
        

    }&
    done
    wait

    for ((j=0;j<$len_gpu;j++))
    do
    {
        cat $hypo_dir/hypo${ckpt_tag}_$j.sorted >> $hypo_dir/hypo${ckpt_tag}.sorted
    }
    done
    # Tokenize.
    perl $tokenizer -l en < $hypo_dir/hypo$ckpt_tag.sorted > $hypo_dir/hypo$ckpt_tag.sorted.tok

    # Get rouge scores
    nohup python3 get_rouge.py --decodes_filename $hypo_dir/hypo$ckpt_tag.sorted.tok --targets_filename cnndm.test.target.tok > $hypo_dir/rouge$ckpt_tag.log 2>&1 &

    test_name=cnndm.test.target

    

!
done
}
done
wait
echo "done decode"


