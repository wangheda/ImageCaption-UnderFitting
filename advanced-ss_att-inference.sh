#!/bin/bash

model_name="show_and_tell_advanced_model_attention_finetune_with_decay"
model="ShowAndTellAdvancedModel"
ckpt=580699
num_processes=1
device=0

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_DIR="${DIR}/model/${model_name}"
IMAGE_DIR="${DIR}/data/ai_challenger_caption_test1_20170923/caption_test1_images_20170923"

CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.inference"

mkdir $OUTPUT_DIR

cd ${DIR}/im2txt

for prefix in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do 
  echo "CUDA_VISIBLE_DEVICES=$device python inference.py \
    --input_file_pattern='${IMAGE_DIR}/${prefix}*.jpg' \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --vocab_file=${DIR}/data/word_counts.txt \
    --output=${OUTPUT_DIR}/part-${prefix}.json \
    --model=${model} \
    --inception_return_tuple=True \
    --use_attention_wrapper=True \
    --attention_mechanism=BahdanauAttention \
    --num_lstm_layers=1 \
    --support_ingraph=True"
done | bash #parallel -j $num_processes

cd ${DIR}

python tools/merge_json_lists.py ${OUTPUT_DIR}/part-?.json > ${OUTPUT_DIR}/all.json

echo output saved to ${OUTPUT_DIR}/all.json
