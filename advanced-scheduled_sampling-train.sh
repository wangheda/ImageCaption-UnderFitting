#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/TFRecord_data"
MODEL_DIR="${DIR}/model"
model=ShowAndTellAdvancedModel

model_dir_name=show_and_tell_advanced_model_ss

cd im2txt && CUDA_VISIBLE_DEVICES=0 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --model=${model} \
  --use_scheduled_sampling=True \
  --scheduled_sampling_method="linear" \
  --scheduled_sampling_starting_rate=0.0 \
  --scheduled_sampling_ending_rate=0.25 \
  --scheduled_sampling_starting_step=0 \
  --scheduled_sampling_ending_step=105000 \
  --initial_learning_rate=1.0 \
  --learning_rate_decay_factor=0.6 \
  --support_ingraph=True \
  --number_of_steps=105000
