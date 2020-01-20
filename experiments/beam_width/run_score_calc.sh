SCORE_DIR="scores_regless"
for dir in ./$SCORE_DIR/*
do
    grep ^H $dir/gen.out | cut -f3- > $dir/gen.out.sys
    grep ^T $dir/gen.out | cut -f2- > $dir/gen.out.ref
    grep ^S $dir/gen.out | cut -f2- > $dir/gen.out.in
    CUDA_VISIBLE_DEVICES=3 fairseq-score --sys $dir/gen.out.sys --ref $dir/gen.out.ref --sentence-bleu > $dir/score.out
done
