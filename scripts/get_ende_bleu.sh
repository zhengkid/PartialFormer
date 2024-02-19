#!/bin/bash

decodes_file=$1
reference_file=$reference
sed -i s'/@@ //g' $decodes_file
#detokenize the decodes file to format the manner to do tokenize
perl $detokenizer -l de < $decodes_file > $decodes_file.dtk
#replace unicode
perl $replace_unicode_punctuation -l de < $decodes_file.dtk > $decodes_file.dtk.punc
#tokenize the decodes file by moses tokenizer.perl
perl $tokenizer -l de < $decodes_file.dtk.punc > $decodes_file.dtk.punc.tok
#"rich-text format" --> rich ##AT##-##AT## text format.
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.dtk.punc.tok > $decodes_file.dtk.punc.tok.atat
#convert all quot '„' to $quot
#cmd="python -u $conver_quot --i=$decodes_file.dtk.punc.tok.atat $decodes_file.dtk.punc.tok.atat.quot"
#cp $decodes_file.dtk.punc.tok.atat $decodes_file.dtk.punc.tok.atat.quot
#sed -i s'/„/\&quot;/g' $decodes_file.dtk.punc.tok.atat.quot
#compute the bleu score
perl $multi_bleu $reference_file < $decodes_file.dtk.punc.tok.atat
