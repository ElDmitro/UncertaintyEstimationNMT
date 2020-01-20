for i in {1..9}
do
	echo $i
	CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/iwslt14.tokenized.de-en \
    		--path mtransformer/checkpoints/checkpoint_best.pt \
    		--batch-size 64 --beam $i --remove-bpe --diverse-beam-strength 0 > mtransformer/scores_regless/w0$i/gen.out
done

for i in {10..50..10}
do
	echo $i
	CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/iwslt14.tokenized.de-en \
    		--path mtransformer/checkpoints/checkpoint_best.pt \
    		--batch-size 16 --beam $i --remove-bpe --diverse-beam-strength 0 > mtransformer/scores_regless/w$i/gen.out
done

