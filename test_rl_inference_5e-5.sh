#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

model_name="show_and_tell_advanced_model_attention_finetune_with_decay_da_rl_lr0.1"
num_processes=1
gpu_fraction=1
device=1
model=ShowAndTellAdvancedModel
ckpt=840423

MODEL_DIR="/home/share/limiao/ImageCaption/model/x39_rl_lr5e-5_d308"
IMAGE_DIR="${DIR}/data/ai_challenger_caption_test1_20170923/caption_test1_images_20170923"
CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.inference"

mkdir $OUTPUT_DIR

cd ${DIR}/im2txt
if [ ! -f ${OUTPUT_DIR}/out.json ]; then
  CUDA_VISIBLE_DEVICES=$device python inference.py \
    --input_file_pattern="${IMAGE_DIR}/${prefix}*.jpg" \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --vocab_file=${DIR}/data/word_counts.txt \
    --output=${OUTPUT_DIR}/out.json \
    --model=${model} \
    --inception_return_tuple=True \
    --use_attention_wrapper=True \
    --attention_mechanism=BahdanauAttention \
    --num_lstm_layers=1 \
    --support_ingraph=True \
    --gpu_memory_fraction=$gpu_fraction
  echo output saved to ${OUTPUT_DIR}/out.json
fi

