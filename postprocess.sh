source_lang=de_DE
target_lang=en_XX
dir=path-to-model #，模型的路径
cat $dir/${source_lang}_${target_lang}.txt | grep -P "^H" |sort -V |cut -f 3-  > $dir/${source_lang}_${target_lang}.hyp
cat $dir/${source_lang}_${target_lang}.txt | grep -P "^T" |sort -V |cut -f 2-  > $dir/${source_lang}_${target_lang}.ref
cat $dir/${source_lang}_${target_lang}.txt | grep -P "^S" |sort -V |cut -f 2-  > $dir/${source_lang}_${target_lang}.src
sacrebleu -tok 'none' -s 'none' $dir/${source_lang}_${target_lang}.ref < $dir/${source_lang}_${target_lang}.hyp
