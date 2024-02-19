#!/usr/bin/bash
set -e

model_root_dir=checkpoints

# set task
task=cnndm-10k
#task=wmt-en2de
# set tag
model_dir_tag=pre-6-gsan

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt
#checkpoint=checkpoint22.pt

ensemble=5

gpu=1

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation.log

if [ -n "$cpu" ]; then
        use_cpu=--cpu
fi

export CUDA_VISIBLE_DEVICES=$gpu

python3 fairseq_cli/generate.py \
data-bin/$task \
--path $model_dir/$checkpoint \
--gen-subset test \
--truncate-source \
--batch-size 128 \
--lenpen 2.0 \
--min-len 55 \
--max-len-b 140 \
--max-source-positions 500 \
--beam 4 \
--no-repeat-ngram-size 3 \
--remove-bpe \
--quiet \
--output $model_dir/hypo.txt | tee $model_dir/decoder.log

python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted

test_name=cnndm.test.target

# Tokenize.
perl $tokenizer -l en < $model_dir/hypo.sorted > $model_dir/hypo.sorted.tok

# Get rouge scores
python3 get_rouge.py --decodes_filename $model_dir/hypo.sorted.tok --targets_filename cnndm.test.target.tok
