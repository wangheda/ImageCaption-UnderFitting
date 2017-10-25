#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/TFRecord_data"
MODEL_DIR="${DIR}/model"

model=ShowAndTellAdvancedModel
model_dir_name=show_and_tell_advanced_model_ss_finetune_with_decay
original_model_dir_name=show_and_tell_advanced_model_ss
start_ckpt=105000

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
  --initial_learning_rate=1.0 \
  --learning_rate_decay_factor=0.6 \
  --train_inception_with_decay=True \
  --support_ingraph=True \
  --use_scheduled_sampling=True \
  --scheduled_sampling_method="linear" \
  --scheduled_sampling_starting_rate=0.0 \
  --scheduled_sampling_ending_rate=0.5 \
  --scheduled_sampling_starting_step=105000 \
  --scheduled_sampling_ending_step=600000 \
  --number_of_steps=600000
