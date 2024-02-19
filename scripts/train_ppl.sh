train_log=$1

grep " INFO | train | epoch " $train_log > grep_log

awk -F '|' '{ print $3, $4, $7, $12 }' grep_log > view_train_ppl.log

grep "ppl" view_train_ppl.log
#rm grep_log
