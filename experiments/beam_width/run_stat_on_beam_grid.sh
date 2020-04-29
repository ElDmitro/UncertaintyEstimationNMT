#! /bin/bash/
GPU_DEVICE_ID=1

DATA_PATH="data-bin/iwslt14.tokenized.de-en/"
MODEL_PATH="mtransformer/de-en"
STATS_PATH="stats_softmax"
BPE_PATH="data/iwslt14.tokenized.de-en/code"

SRC_LANG="de"
TGT_LANG="en"
MAX_SENTENCES=3500

for i in 2 # 5 9 20 50
do
	OUT_PATH="${MODEL_PATH}/${STATS_PATH}/w_${i}"
	echo $OUT_PATH
	if [ ! -d ./$OUT_PATH ]; then
		mkdir ./$OUT_PATH
	fi
	CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} \
	python run_stats_collector.py --model-path ${MODEL_PATH} \
	--checkpoint-paths model1.pt model2.pt model3.pt model4.pt model5.pt \
	--data-path ${DATA_PATH} \
	--bpe-path ${BPE_PATH} \
	--beam ${i} --lenpen 0 --shared-bpe \
	--src-lang ${SRC_LANG} --tgt-lang ${TGT_LANG} \
	--log-dir ${OUT_PATH} --device-id ${GPU_DEVICE_ID} \
	--max-sentences ${MAX_SENTENCES}
done
