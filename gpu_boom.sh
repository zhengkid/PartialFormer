device=0,1,2,3,4,5,6,7
task=wmt-en2de
# must set this tag
tag=gpu_boom_fp32


if [ $task == "wmt-en2de" ]; then
        arch=transformer_t2t_wmt_en_de
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=0
        lr=0.002
        warmup=16000
        max_tokens=512
        update_freq=1
        weight_decay=0.0
        keep_last_epochs=10
        max_epoch=9999
        max_update=
        reset_optimizer=0
        data_dir=google
        src_lang=en
        tgt_lang=de
elif [ $task == "wmt-en2de-32K" ]; then
        arch=transformer_wmt_en_de
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.0007
        warmup=4000
        max_tokens=4096
        update_freq=1
        weight_decay=0.0
        keep_last_epochs=10
        max_epoch=21
        max_update=100000
        reset_optimizer=0
        data_dir=wmt-390-32k
        src_lang=en
        tgt_lang=de
elif [ $task == "best-paper-11k" ]; then
        arch=transformer_wmt_en_de_big
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.0005
        warmup=4000
        max_tokens=9600
        update_freq=8
        weight_decay=0.0
        keep_last_epochs=10
        max_epoch=100
        max_update=
        reset_optimizer=0
        data_dir=best-paper-11k
        src_lang=en
        tgt_lang=de
elif [ $task == "wmt14-en2de-volt-wo-join" ]; then
        arch=transformer_t2t_wmt_en_de
        share_embedding=0
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.001
        warmup=8000
        max_tokens=4096
        update_freq=2
        weight_decay=0.0
        keep_last_epochs=20
        max_epoch=30
        max_update=
        reset_optimizer=0
        data_dir=google-best-paper
        src_lang=en
        tgt_lang=de
elif [ $task == "iwslt-de2en" ]; then
        arch=identity_transformer_t2t_iwslt_de_en
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.0015
        warmup=8000
        max_tokens=4096
        update_freq=1
        weight_decay=0.0001
        keep_last_epochs=20
        max_epoch=51
        max_update=
        reset_optimizer=0
        data_dir=iwslt-14-de2en
        src_lang=de
        tgt_lang=en
elif [ $task == "wmt-en2ro-final" ]; then
        arch=identity_transformer_t2t_wmt_en_de
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.001
        warmup=8000
        max_tokens=4096
        update_freq=1
        weight_decay=0.0
        keep_last_epochs=15
        max_epoch=20
        max_update=
        reset_optimizer=0
        data_dir=wmt16.en-ro.bpe20k
        src_lang=en
        tgt_lang=ro
elif [ $task == "ldc" ]; then
        arch=runge_kutta_relative_transformer_t2t_wmt_en_de
        share_embedding=0
        share_decoder_input_output_embed=1
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.002
        warmup=8000
        max_tokens=2048
        update_freq=4
        weight_decay=0.0
        keep_last_epochs=15
        max_epoch=20
        max_update=
        reset_optimizer=0
        data_dir=LDC_180W
        src_lang=zh
        tgt_lang=en
elif [ $task == "wmt-en2fr" ]; then
        arch=group_share_relative_transformer_t2t_wmt_en_de_big
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        reset_optimizer=0
        fp16=1
        lr=0.002
        warmup=16000
        max_tokens=4096
        update_freq=8
        weight_decay=0.0
        keep_last_epochs=20
        max_epoch=20
        max_update=
        data_dir=wmt_en_fr_joint_bpe
        src_lang=en
        tgt_lang=fr
elif [ $task == "wmt-en2zh" ]; then
        arch=runge_kutta_relative_transformer_t2t_wmt_en_de_big
        share_embedding=0
        share_decoder_input_output_embed=1
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.002
        warmup=16000
        max_tokens=4096
        update_freq=8
        weight_decay=0.0
        keep_last_epochs=10
        max_epoch=20
        max_update=
        data_dir=wmt-en2zh-v2
        src_lang=en
        tgt_lang=zh
elif [ $task == "wmt-zh2en" ]; then
        arch=runge_kutta_relative_transformer_t2t_wmt_en_de
        share_embedding=0
        share_decoder_input_output_embed=1
        criterion=label_smoothed_cross_entropy
        reset_optimizer=0
        fp16=1
        lr=0.002
        warmup=16000
        max_tokens=4096
        update_freq=2
        weight_decay=0.0
        keep_last_epochs=15
        max_epoch=17
        max_update=
        data_dir=WMT21-zh2en-z6
        src_lang=zh
        tgt_lang=en
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

cmd="python3 -u train.py data-bin/$data_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --optimizer adam --clip-norm 0.0
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --lr $lr --min-lr 1e-09
  --weight-decay $weight_decay
  --criterion $criterion --label-smoothing 0.1
  --max-tokens $max_tokens
  --update-freq $update_freq
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d 
  --seed 1
  --encoder-layers 6
  --task translation
  --save-dir $save_dir
  --keep-last-epochs $keep_last_epochs
  --tensorboard-logdir $save_dir"


adam_betas="'(0.9, 0.997)'"
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
if [ $reset_optimizer -eq 1 ]; then
cmd=${cmd}" --reset-optimizer "
fi

#echo $cmd
#eval $cmd
#cmd=$(eval $cmd)
#nohup $cmd exec 1> $save_dir/train.log exec 2>&1 &
#tail -f $save_dir/train.log

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log

