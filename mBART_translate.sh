#!/usr/bin/bash
set -e

model_root_dir=checkpoints

# set task
#task=codeSum
#task=wmt14-en2de-volt-wo-join
#task=wmt-en2ro-final
task=iwslt-de2en
#task=best-paper-11k
#task=wmt-volt-15k
# set tag
model_dir_tag=multi_scale_double_6

#model_dir_tag=baseline

#model_dir_tag=base40_rpr_only_k_max8
#model_dir_tag=base6_rpr_decoder3
#model_dir_tag=fusion48_rpr_48_0.002
#model_dir_tag=finetune_stochastic40_rpr_p4
# set device
gpu=0
cpu=

# data set
who=test
if [ $task == "iwslt-de2en" ]; then
        data_dir=iwslt-14-de2en
        ensemble=5
        batch_size=64
        fairseq_task=translation
        beam=8
        length_penalty=1.0
        src_lang=de
        tgt_lang=en
        sacrebleu_set=wmt14/full
elif [ $task == "iwslt-fa2en" ]; then
        data_dir=iwslt-14-fa2en
        ensemble=5
        batch_size=128
        fairseq_task=translation
        beam=8
        length_penalty=1.0
        src_lang=fa
        tgt_lang=en
        sacrebleu_set=wmt14/full
elif [ $task == "iwslt-en2de" ]; then
        data_dir=iwslt-14-en2de
        ensemble=5
        batch_size=128
        fairseq_task=translation
        beam=8
        length_penalty=1.0
        src_lang=en
        tgt_lang=de
        sacrebleu_set=wmt14/full
elif [ $task == "iwslt-ro2en" ]; then
        data_dir=iwslt-14-ro2en
        ensemble=5
        batch_size=128
        fairseq_task=translation
        beam=8
        length_penalty=1.0
        src_lang=ro
        tgt_lang=en
        sacrebleu_set=wmt14/full
elif [ $task == "iwslt-es2en" ]; then
        data_dir=iwslt-14-es2en
        ensemble=5
        batch_size=128
        fairseq_task=translation
        beam=8
        length_penalty=1.0
        src_lang=es
        tgt_lang=en
        sacrebleu_set=wmt14/full
elif [ $task == "iwslt-en2es" ]; then
        data_dir=iwslt-14-en2es
        ensemble=5
        batch_size=128
        fairseq_task=translation
        beam=8
        length_penalty=1.0
        src_lang=en
        tgt_lang=es
        sacrebleu_set=wmt14/full
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
--remove-bpe | tee $output
bash scripts/compound_split_bleu.sh $output | tee $compound
python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted

if [ $data_dir == "google" ]; then
	sh $get_ende_bleu $model_dir/hypo.sorted
	perl $detokenizer -l de < $model_dir/hypo.sorted > $model_dir/hypo.dtk
elif [ $data_dir == "wmt17_en_de" ]; then
        sh $get_ende_bleu $model_dir/hypo.sorted
        perl $detokenizer -l de < $model_dir/hypo.sorted > $model_dir/hypo.dtk
elif [ $data_dir == "best-paper-11k" ]; then
        sh scripts/compound_split_bleu.sh $output
        #sh scripts/sacrebleu.sh wmt14/full en de $output
elif [ $data_dir == "best-paper-11k-phrase" ]; then
        sh scripts/compound_split_bleu.sh $output
elif [ $data_dir == "google-10K" ]; then
        sh $get_ende_bleu $model_dir/hypo.sorted
        perl $detokenizer -l de < $model_dir/hypo.sorted > $model_dir/hypo.dtk
elif [ $data_dir == "google-best-paper" ]; then
        #sh $get_ende_bleu $model_dir/hypo.sorted
        #perl $detokenizer -l de < $model_dir/hypo.sorted > $model_dir/hypo.dtk
        sh scripts/compound_split_bleu.sh $output
        #sh scripts/sacrebleu.sh wmt14/full en de $output
elif [ $data_dir == "best-paper-32k" ]; then
        #sh $get_ende_bleu $model_dir/hypo.sorted
        #perl $detokenizer -l de < $model_dir/hypo.sorted > $model_dir/hypo.dtk
        sh scripts/compound_split_bleu.sh $output
        #sh scripts/sacrebleu.sh wmt14/full en de $output
elif [ $data_dir == "wmt-volt-15k" ]; then
        #sh $get_ende_bleu $model_dir/hypo.sorted
        #perl $detokenizer -l de < $model_dir/hypo.sorted > $model_dir/hypo.dtk
        sh scripts/compound_split_bleu.sh $output
        #sh scripts/sacrebleu.sh wmt14/full en de $output
elif [ $data_dir == "google-10k-tree-new" ]; then
        sh $get_ende_bleu $model_dir/hypo.sorted
        perl $detokenizer -l de < $model_dir/hypo.sorted > $model_dir/hypo.dtk
elif [ $data_dir == "macaron" ]; then
        sh $get_ende_bleu $model_dir/hypo.sorted
fi




#if [ $sacrebleu_set == "wmt14/full" ]; then

#        echo -e "\n>> BLEU-13a"
#        cat $model_dir/hypo.dtk | sacrebleu ../reference/en-de.de -tok 13a

#        echo -e "\n>> BLEU-intl"
#        cat $model_dir/hypo.dtk | sacrebleu ../reference/en-de.de -tok intl
#fi

