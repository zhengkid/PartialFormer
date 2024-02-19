train_log=$1
grep "valid on" $train_log > grep_log

awk -F '|' '{ print $3, $4, $8, $12 }' grep_log > view.log

rm grep_log

grep "ppl" view.log
