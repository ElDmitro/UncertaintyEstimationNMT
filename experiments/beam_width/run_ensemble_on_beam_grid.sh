#! /bin/bash/
GPU_DEVICE_ID=2

DATA_PATH="data-bin/iwslt14.tokenized.de-en"

MODEL_PATH="mtransformer/de-en"
SCORES_PATH="scores_regless"

COMPLETE_PATH=""
for CHECKPOINT in "model1.pt" "model2.pt" "model3.pt" "model4.pt" "model5.pt"
do
    COMPLETE_PATH=$COMPLETE_PATH":"$MODEL_PATH"/$CHECKPOINT"
done
COMPLETE_PATH=${COMPLETE_PATH:1}
echo $COMPLETE_PATH

for i in {4..9..3}
do
	OUT_PATH="${MODEL_PATH}/${SCORES_PATH}/w0${i}"
	echo $OUT_PATH
	if [ ! -d ./$OUT_PATH ]; then
		mkdir ./$OUT_PATH
	fi
	CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID fairseq-generate ./$DATA_PATH \
    		--path ./$COMPLETE_PATH \
    		--batch-size 16 --beam $i --remove-bpe --diverse-beam-strength 0 \
		--lenpen 0 > ./$OUT_PATH/gen.out
done

for i in {10..50..10}
do
	OUT_PATH="${MODEL_PATH}/${SCORES_PATH}/w${i}"
	echo $OUT_PATH
	if [ ! -d ./$OUT_PATH ]; then
		mkdir ./$OUT_PATH
	fi
	CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID fairseq-generate ./$DATA_PATH \
    		--path ./$COMPLETE_PATH \
    		--batch-size 8 --beam $i --remove-bpe --diverse-beam-strength 0 \
		--lenpen 0 > ./$OUT_PATH/gen.out
done

