#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/TFRecord_data"
MODEL_DIR="${DIR}/model"

model=ShowAndTellInGraphModel
model_dir_name=show_and_tell_in_graph_model_adam_finetune
original_model_dir_name=show_and_tell_in_graph_model_adam
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
  --train_inception=True \
  --train_inception_learning_rate=0.0003 \
  --support_ingraph=True \
  --exclude_variable_patterns='OptimizeLoss/InceptionV3/.+/Adam.*' \
  --optimizer=Adam \
  --clip_gradients=1.0 \
  --number_of_steps=210000
