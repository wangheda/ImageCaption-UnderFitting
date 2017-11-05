#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/TFRecord_data"
MODEL_DIR="${DIR}/model"

model=SemanticAttentionModel
model_dir_name=semantic_attention_model_join
original_model_dir_name=semantic_attention_model_attr_only
start_ckpt=180973

# copy the starting checkpoint
if [ ! -d ${MODEL_DIR}/${model_dir_name} ]; then
  mkdir -p ${MODEL_DIR}/${model_dir_name}
  cp ${MODEL_DIR}/${original_model_dir_name}/model.ckpt-${start_ckpt}.* ${MODEL_DIR}/${model_dir_name}/
  echo "model_checkpoint_path: \"${MODEL_DIR}/${model_dir_name}/model.ckpt-${start_ckpt}\"" > ${MODEL_DIR}/${model_dir_name}/checkpoint
fi

cd im2txt && CUDA_VISIBLE_DEVICES=0 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --model=${model} \
  --support_ingraph=True \
  --number_of_steps=800000 \
  --initial_learning_rate=4.6 \
  --learning_rate_decay_factor=0.6 \
  --attributes_top_k=15 \
  --train_inception_with_decay=True \
  --vocab_file="${DIR}/data/word_counts.txt" \
  --attributes_file="${DIR}/data/attributes.txt"

