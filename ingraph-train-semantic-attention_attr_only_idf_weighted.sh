#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/TFRecord_data"
MODEL_DIR="${DIR}/model"
model=SemanticAttentionModel

model_dir_name=semantic_attention_model_attr_only_idf_weighted

cd im2txt && CUDA_VISIBLE_DEVICES=1 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --model=${model} \
  --support_ingraph=True \
  --number_of_steps=200000 \
  --only_attributes_loss=True \
  --use_idf_weighted_attribute_loss=True \
  --vocab_file="${DIR}/data/word_counts.txt" \
  --attributes_file="${DIR}/data/attributes.txt" \
  --word_idf_file="${DIR}/data/word_idf.txt"
