#! /bin/bash

export PYTHONPATH=/mnt/nfs/scratch1/dthai/fast-coref/src/:$PYTHONPATH

data_dir=/mnt/nfs/scratch1/sdhuliawala/dpr_sentence_splits_genre_tagged_final
out_dir=/mnt/nfs/scratch1/dthai/fast-coref/tagged_wiki
filename=file_0-1-genre.jsonl
encoder_name=shtoshni/longformer_coreference_joint
model_path=/mnt/nfs/scratch1/dthai/fast-coref/

run_cmd="python main.py  \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --filename $filename \
    --model_path $model_path \
    --encoder_name $encoder_name \
    --use_wandb 0 \
"

echo $run_cmd
eval $run_cmd