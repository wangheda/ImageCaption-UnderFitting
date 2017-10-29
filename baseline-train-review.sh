#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
#TFRECORD_DIR="${DIR}/data/TFRecord_data"
INCEPTION_CHECKPOINT="/media/diskb/x39_share/ImageCaption/inception/inception_v3.ckpt"
TFRECORD_DIR="/media/diskb/x39_share/ImageCaption/data/TFRecord_data"
MODEL_DIR="${DIR}/model"

model_dir_name=review_model
model=ReviewnetworkModel

cd im2txt && CUDA_VISIBLE_DEVICES=0 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --model=${model} \
  --train_inception_with_decay=True \
  --number_of_steps=840000
