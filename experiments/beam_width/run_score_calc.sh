GPU_DEVICE_ID=3

MODEL_DIR="wmt14.en-fr.fconv-py"
SCORE_DIR="scores_regless"

for dir in ./$MODEL_DIR/$SCORE_DIR/*
do
    grep ^H $dir/gen.out | cut -f3- > $dir/gen.out.sys
    grep ^T $dir/gen.out | cut -f2- > $dir/gen.out.ref
    grep ^S $dir/gen.out | cut -f2- > $dir/gen.out.in
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID fairseq-score --sys $dir/gen.out.sys --ref $dir/gen.out.ref --sentence-bleu --diverse-beam-strength 0 > $dir/score.out
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID fairseq-score --sys $dir/gen.out.sys --ref $dir/gen.out.ref --diverse-beam-strength 0 > $dir/total.out
done
