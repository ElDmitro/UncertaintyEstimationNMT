GPU_DEVICE_ID=3

DATA_PATH="data-bin/wmt14.en-fr.newstest2014"

MODEL_PATH="wmt14.en-fr.fconv-py"
CHECKPOINT_PATH="model.pt"
SCORES_PATH="scores_regless"

for i in {1..9}
do
	OUT_PATH="${MODEL_PATH}/${SCORES_PATH}/w0${i}"
	echo $OUT_PATH
	if [ ! -d ./$OUT_PATH ]; then
		mkdir ./$OUT_PATH
	fi
	CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID fairseq-generate ./$DATA_PATH \
    		--path ./$MODEL_PATH/$CHECKPOINT_PATH \
    		--batch-size 64 --beam $i --remove-bpe --diverse-beam-strength 0 > ./$OUT_PATH/gen.out
done

for i in {10..50..10}
do
	OUT_PATH="${MODEL_PATH}/${SCORES_PATH}/w${i}"
	echo $OUT_PATH
	if [ ! -d ./$OUT_PATH ]; then
		mkdir ./$OUT_PATH
	fi
	CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID fairseq-generate ./$DATA_PATH \
    		--path ./$MODEL_PATH/$CHECKPOINT_PATH \
    		--batch-size 16 --beam $i --remove-bpe --diverse-beam-strength 0 > ./$OUT_PATH/gen.out
done

