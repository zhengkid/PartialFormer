# WMT16 EN-RO
cd data
#mkdir wmt16.en-ro
cd wmt16.en-ro
#gdown https://drive.google.com/uc?id=1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl
tar -xvf wmt16.tar
mv wmt16/en-ro/train/corpus.bpe.en train.en
mv wmt16/en-ro/train/corpus.bpe.ro train.ro
mv wmt16/en-ro/dev/dev.bpe.en valid.en
mv wmt16/en-ro/dev/dev.bpe.ro valid.ro
mv wmt16/en-ro/test/test.bpe.en test.en
mv wmt16/en-ro/test/test.bpe.ro test.ro
#rm wmt16.tar.gz
#rm -r wmt16
cd ..
python3 preprocess.py --source-lang en --target-lang ro --trainpref data/wmt16.en-ro/train --validpref data/wmt16.en-ro/valid --testpref data/wmt16.en-ro/test --destdir data-bin/wmt16.en-ro --joined-dictionary --workers 8 --nwordssrc 40000 --nwordstgt 40000
python3 preprocess.py --source-lang ro --target-lang en --trainpref data/wmt16.en-ro/train --validpref data/wmt16.en-ro/valid --testpref data/wmt16.en-ro/test --destdir data-bin/wmt16.ro-en --joined-dictionary --workers 8 --nwordssrc 40000 --nwordstgt 40000
