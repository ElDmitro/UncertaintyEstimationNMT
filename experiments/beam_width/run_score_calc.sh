GPU_DEVICE_ID=1

MODEL_DIR="wmt14.en-fr.fconv-py"
SCORE_DIR="scores_regless"

for dir in ./$MODEL_DIR/$SCORE_DIR/*
do
    echo $dir
    grep ^H $dir/gen.out | cut -f3- > $dir/gen.out.sys
    grep ^T $dir/gen.out | cut -f2- > $dir/gen.out.ref
    grep ^S $dir/gen.out | cut -f2- > $dir/gen.out.in
    grep ^S $dir/gen.out | cut -f1 | cut -f2 -d '-' > $dir/idxs.out
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID fairseq-score --sys $dir/gen.out.sys --ref $dir/gen.out.ref --sentence-bleu > $dir/score.out
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID fairseq-score --sys $dir/gen.out.sys --ref $dir/gen.out.ref > $dir/total.out
done
