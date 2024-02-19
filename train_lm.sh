device=0,1,2,3,4,5,6,7
task=wiki
tag=pre_8_partialformer_A_G_MHSA_A_L_MHSA_HEAD_8_EXPAND_HEAD_16_DIM_1024_Dropout_0.3
if [ $task == "wiki" ]; then
        arch=pa_transformer_lm_wiki103
        lr=0.0001
        warmup=16000
        max_tokens=2048
        tokens_per_sample=${max_tokens}
        update_freq=4
        weight_decay=0
        keep_last_epochs=5
        criterion=adaptive_loss
        max_epoch=
        max_update=286000
        data_dir=wikitext-103
        fp16=1
else
        echo "unknown task=$task"
        exit
fi

save_dir=checkpoints/$task/$tag


if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
cp ${BASH_SOURCE[0]} $save_dir/train.sh

#gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`
gpu_num=8

cmd="python3 -u train.py data-bin/${data_dir}
      --task language_modeling
      --max-lr 1.0
      --t-mult 2
      --lr-period-updates 270000
      --lr-scheduler cosine
      --lr-shrink 0.75
      --warmup-init-lr 1e-07
      --min-lr 1e-09
      --optimizer nag
      --clip-norm 0.1
      --seed 1
      --sample-break-mode none
      --skip-invalid-size-inputs-valid-test
      --ddp-backend=no_c10d
      --dropout 0.3
      --attention-dropout 0.1
      --relu-dropout 0.1
      --decoder-layers 8
      --decoder-attention-heads 8
      --decoder-global-heads 8 
      --decoder-embed-dim 1024
      --decoder-hidden-ratio 4
      --standard-mhsa
      --use-hdp
      --decoder-expand-heads 16
      --add-dropout
      --decoder-global
      --log-interval 100
      --criterion ${criterion}
      --weight-decay $weight_decay
      --distributed-world-size $gpu_num
      --keep-last-epochs $keep_last_epochs
      --tensorboard-logdir $save_dir
      --save-dir ${save_dir}
      --arch ${arch}
      --warmup-updates ${warmup}
      --max-tokens ${max_tokens}
      --update-freq ${update_freq}
      --tokens-per-sample ${tokens_per_sample}
      --lr ${lr}"

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi
if [ -n "$max_epoch" ]; then
cmd=${cmd}" --max-epoch "${max_epoch}
fi
if [ -n "$max_update" ]; then
cmd=${cmd}" --max-update "${max_update}
fi

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
