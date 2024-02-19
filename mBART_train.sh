device=0,1,2,3,4,5,6,7

task=wmt-en2de-sp

# must set this tag
tag=mBart-init-transformer-baseline
if [ $task == "wmt-en2de-sp" ]; then
        arch=mbart_large
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=3e-05
        warmup=2500
        dropout=0.3
        max_tokens=2048
        update_freq=1
        weight_decay=0.000
        keep_last_epochs=20
        max_epoch=25
        max_update=300000
        data_dir=wmt_16_en_De-sp
        lang_list=data-bin/lang_list_25
        lang_pairs=en_XX-de_DE
        pretrained_model=../mbart.cc25/model.pt

else
        echo "unknown task=$task"
        exit
fi

save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=8

cmd="python -u train.py data-bin/$data_dir
  --finetune-from-model $pretrained_model
  --encoder-normalize-before --decoder-normalize-before
  --distributed-world-size $gpu_num
  --arch $arch
  --layernorm-embedding
  --task translation_multi_simple_epoch
  --sampling-method "temperature" 
  --sampling-temperature 1.5 
  --encoder-langtok "src" 
  --decoder-langtok 
  --lang-dict "$lang_list" 
  --lang-pairs "$lang_pairs" 
  --optimizer adam --adam-eps 1e-06
  --lr-scheduler inverse_sqrt
  --lr $lr 
  --weight-decay $weight_decay
  --warmup-updates $warmup
  --criterion $criterion --label-smoothing 0.2
  --max-tokens $max_tokens
  --update-freq $update_freq
  --no-progress-bar
  --max-update $max_update
  --log-interval 100
  --seed 222
  --ddp-backend no_c10d 
  --save-dir $save_dir
  --keep-last-epochs $keep_last_epochs
  --tensorboard-logdir $save_dir
  --attention-dropout 0.1"

adam_betas="'(0.9, 0.98)'"
cmd=${cmd}" --adam-betas "${adam_betas}
if [ $share_embedding -eq 1 ]; then
cmd=${cmd}" --share-all-embeddings "
fi
if [ $share_decoder_input_output_embed -eq 1 ]; then
cmd=${cmd}" --share-decoder-input-output-embed "
fi
if [ -n "$max_epoch" ]; then
cmd=${cmd}" --max-epoch "${max_epoch}
fi
if [ -n "$max_update" ]; then
cmd=${cmd}" --max-update "${max_update}
fi
if [ -n "$dropout" ]; then
cmd=${cmd}" --dropout "${dropout}
fi
if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi

#echo $cmd
#eval $cmd
#cmd=$(eval $cmd)
#nohup $cmd exec 1> $save_dir/train.log exec 2>&1 &
#tail -f $save_dir/train.log

# kill the running system
ps -aux | grep "finetune" | grep -v grep | awk '{print $2}' | xargs kill -9
ps -aux | grep "fairseq" | grep -v grep | awk '{print $2}' | xargs kill -9

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log

cd ../GODIVA
bash run.sh -a do_train -ltbs 1 -lebs 1 -cf ./config/GPUBoom.py -m dist -p local
