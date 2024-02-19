#!/usr/bin/bash
set -e

model_root_dir=checkpoints

# set task
#task=iwslt-de2en
task=wmt14-en2de-volt-wo-join
# set tag
model_dir_tag=post_norm_rpr_big

#model_dir_tag=base40_rpr_only_k_max8
#model_dir_tag=base6_rpr_decoder3
#model_dir_tag=fusion48_rpr_48_0.002
#model_dir_tag=finetune_stochastic40_rpr_p4
# set device
gpu=(0 1 2 3 4 5 6 7)
cpu=

# data set
who=test
upper_bound=20
lower_bound=10
start_avg_num=5

if [ $task == "wmt-en2de" ]; then
        data_dir=google
        ensemble=5
        batch_size=64
        beam=8
        length_penalty=0.6
        src_lang=en
        tgt_lang=de
        sacrebleu_set=
elif [ $task == "wmt14-en2de-volt-wo-join" ]; then
        data_dir=google-best-paper
        ensemble=$1
        fairseq_task=dp_tree_group_phrase_translation
        batch_size=64
        beam=4
        length_penalty=0.6
        src_lang=en
        tgt_lang=de
        sacrebleu_set=
elif [ $task == "wmt-en2ro-final" ]; then
        data_dir=wmt16.en-ro.bpe20k
        ensemble=5
        batch_size=128
        beam=5
        fairseq_task=dp_tree_group_phrase_translation
        length_penalty=1.3
        src_lang=en
        tgt_lang=ro
        sacrebleu_set=
elif [ $task == "wmt-zh2en" ]; then
        data_dir=wmt-zh2en
        ensemble=
        batch_size=64
        beam=6
        length_penalty=1.3
        src_lang=zh
        tgt_lang=en
        sacrebleu_set=
elif [ $task == "iwslt-de2en" ]; then
        data_dir=iwslt14.tokenized.de-en
        ensemble=5
        batch_size=64
        beam=5
        length_penalty=1.0
        src_lang=de
        tgt_lang=en
        sacrebleu_set=wmt14/full
elif [ $task == "wmt-en2fr" ]; then
        data_dir=wmt_en_fr_joint_bpe
        ensemble=$1
        batch_size=64
        beam=5
        length_penalty=0.8
        src_lang=en
        tgt_lang=fr
        sacrebleu_set=wmt14/full
elif [ $task == "ldc" ]; then
        data_dir=LDC_180W
        ensemble=5
        batch_size=64
        beam=6
        length_penalty=1.3
        src_lang=zh
        tgt_lang=en
        sacrebleu_set=
elif [ $task == "yatrans-en2zh" ]; then
        data_dir=yatrans-en2zh
        ensemble=4
        batch_size=64
        beam=6
        length_penalty=0.7
        src_lang=en
        tgt_lang=zh
        sacrebleu_set=
elif [ $task == "wmt-en2ro" ]; then
        data_dir=wmt-en2ro
        ensemble=$1
        batch_size=128
        beam=5
        length_penalty=1.3
        src_lang=en
        tgt_lang=ro
        sacrebleu_set=

else
        echo "unknown task=$task"
        exit
fi

model_dir=$model_root_dir/$task/$model_dir_tag


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
    python3 fairseq_cli/generate.py \
    data-bin/$data_dir \
    --path $model_dir/ensemble_ckpts/$ckpt \
    --task $fairseq_task \
    --gen-subset $who \
    --fp16 \
    --batch-size $batch_size \
    --beam $beam \
    --lenpen $length_penalty \
    --output $hypo_dir/hypo$ckpt_tag.txt \
    --remove-bpe > $output


    score=`cat $output | grep -o 'BLEU4 = [0-9]\+.[0-9]\+'`
    echo "$ckpt  $score" >> $result
    #python3 rerank.py $hypo_dir/hypo$ckpt_tag.txt $hypo_dir/hypo$ckpt_tag.sorted
    sh scripts/compound_split_bleu.sh $output > $hypo_dir/multibleu.$ckpt_tag.log
    #sh $get_ende_bleu $hypo_dir/hypo$ckpt_tag.sorted > $hypo_dir/multibleu.$ckpt_tag.dtk.log
done
}&
done
wait
echo "done decode"


