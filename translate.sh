model_root_dir=checkpoints

# set task
#task=codeSum
#task=wmt14-en2de-volt-wo-join
task=wmt-en2de
#task=wmt-en2de
#task=best-paper-11k
#task=wmt-volt-15k
# set tag
model_dir_tag=pre_24_PartialFormer

# set device
gpu=0
cpu=

# data set
who=test

if [ $task == "wmt-en2de" ]; then
        data_dir=google
        ensemble=10
        fairseq_task=translation
        batch_size=64
        beam=4
        length_penalty=0.6
        src_lang=en
        tgt_lang=de
        sacrebleu_set=wmt14/full
elif [ $task == "wmt-en2fr" ]; then
        data_dir=wmt_en_fr_bpe32k_v2
        ensemble=5
        batch_size=64
        fairseq_task=translation
        beam=4
        length_penalty=0.8
        src_lang=en
        tgt_lang=fr
        sacrebleu_set=wmt14/full
elif [ $task == "wmt-en2ro-final" ]; then
        data_dir=wmt16.en-ro.bpe20k
        ensemble=5
        batch_size=128
        beam=5
        fairseq_task=translation
        length_penalty=1.3
        src_lang=en
        tgt_lang=ro
        sacrebleu_set=
else
        echo "unknown task=$task"
        exit
fi
model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt
if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/gen.out

if [ -n "$cpu" ]; then
        use_cpu=--cpu
fi

export CUDA_VISIBLE_DEVICES=$gpu

python3 fairseq_cli/generate.py \
data-bin/$data_dir \
--path $model_dir/$checkpoint \
--task $fairseq_task \
--gen-subset $who \
--batch-size $batch_size \
--beam $beam \
--fp16 \
--lenpen $length_penalty \
--output $model_dir/hypo.txt \
--remove-bpe > $output
python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted

if [ $data_dir == "google" ]; then
	sh $get_ende_bleu $model_dir/hypo.sorted
	perl $detokenizer -l de < $model_dir/hypo.sorted > $model_dir/hypo.dtk
fi




#if [ $sacrebleu_set == "wmt14/full" ]; then

#        echo -e "\n>> BLEU-13a"
#        cat $model_dir/hypo.dtk | sacrebleu ../reference/en-de.de -tok 13a

#        echo -e "\n>> BLEU-intl"
#        cat $model_dir/hypo.dtk | sacrebleu ../reference/en-de.de -tok intl
#fi

