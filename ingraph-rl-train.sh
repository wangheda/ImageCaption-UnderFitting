#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/TFRecord_data"
MODEL_DIR="${DIR}/model"
DOCUMENT_FREQUENCY_FILE="${DIR}/data/document_frequency.json"
model=ShowAndTellInGraphModel

model_dir_name=show_and_tell_in_graph_rl_model

cd im2txt && CUDA_VISIBLE_DEVICES=1 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --initial_learning_rate=1.0 \
  --learning_rate_decay_factor=0.6 \
  --model=${model} \
  --support_ingraph=True \
  --rl_train=True \
  --document_frequency_file="${DOCUMENT_FREQUENCY_FILE}" \
  --number_of_steps=105000 \
  --batch_size=30
