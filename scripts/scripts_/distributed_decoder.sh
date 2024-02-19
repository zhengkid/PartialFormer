gpu=8

tag=pre-6-gsan
task=cnndm-10k
mosesdecoder=/mnt/libei/mosesdecoder
ensemble=5
checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "checkpoints/$task/$tag/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs checkpoints/$task/$tag --output checkpoints/$task/$tag/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi


for ((i=0;i<$gpu;i++))
do
{
    export CUDA_VISIBLE_DEVICES=$i
    python3 fairseq_cli/generate.py \
    data-bin/${task} \
    --task cnndm_bpe_phrase_translation \
    --path  checkpoints/$task/$tag/${checkpoint} \
    --gen-subset test$i \
    --truncate-source \
    --batch-size 64 \
    --lenpen 2.0 \
    --fp16 \
    --min-len 55 \
    --max-len-b 140 \
    --max-source-positions 500 \
    --output checkpoints/$task/$tag/hypo_$i.txt \
    --beam 4 \
    --no-repeat-ngram-size 3 \
    --remove-bpe \
    --quiet

    python3 rerank.py checkpoints/$task/$tag/hypo_$i.txt checkpoints/$task/$tag/hypo_$i.sorted

}&
done

wait

for ((i=0;i<$gpu;i++))
do
{
    cat checkpoints/$task/$tag/hypo_$i.sorted >> checkpoints/$task/$tag/hypo.sorted
}
done

# Tokenize.
#perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data-bin/${task}/${test_name} > data-bin/${task}/${test_name}.tok
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < checkpoints/$task/$tag/hypo.sorted > checkpoints/$task/$tag/hypo.sorted.tok

# Get rouge scores
python3 get_rouge.py --decodes_filename checkpoints/$task/$tag/hypo.sorted.tok --targets_filename cnndm.test.target.tok > checkpoints/$task/$tag/rough.log


