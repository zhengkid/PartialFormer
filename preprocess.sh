src=en
tgt=de
TEXT=/home/v-lbei/fairseq-msra/examples/translation/wmt17_en_de
tag=wmt_en_de_bpe32k
output=data-bin/$tag
python3 preprocess.py --source-lang $src --target-lang $tgt --trainpref $TEXT/train --validpref $TEXT/valid  --testpref $TEXT/test  --destdir $output --nwordssrc 32768 --nwordstgt 32768 --joined-dictionary --workers 32

