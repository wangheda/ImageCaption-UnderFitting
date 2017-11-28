#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/Loc_TFRecord_data"
MODEL_DIR="${DIR}/model"
DOCUMENT_FREQUENCY_FILE="${DIR}/data/document_frequency.json"

model=ShowAndTellAdvancedModel
model_dir_name=show_and_tell_advanced_model_visual_attention_rl_adam_lr1e-5_decay
original_model_dir_name=show_and_tell_advanced_model_attention_finetune_with_decay_da
start_ckpt=640713

# copy the starting checkpoint
if [ ! -d ${MODEL_DIR}/${model_dir_name} ]; then
  mkdir -p ${MODEL_DIR}/${model_dir_name}
  cp ${MODEL_DIR}/${original_model_dir_name}/model.ckpt-${start_ckpt}.* ${MODEL_DIR}/${model_dir_name}/
  echo "model_checkpoint_path: \"${MODEL_DIR}/${model_dir_name}/model.ckpt-${start_ckpt}\"" > ${MODEL_DIR}/${model_dir_name}/checkpoint
fi

cd im2txt && CUDA_VISIBLE_DEVICES=1 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --model=${model} \
  --initial_learning_rate=0.00278 \
  --learning_rate_decay_factor=0.6 \
  --inception_return_tuple=True \
  --use_scheduled_sampling=False \
  --use_attention_wrapper=True \
  --attention_mechanism=BahdanauAttention \
  --num_lstm_layers=1 \
  --support_ingraph=True \
  --support_flip=True \
  --batch_size=30 \
  --num_examples_per_epoch=210000 \
  --train_inception_with_decay=False \
  --number_of_steps=900000 \
  --rl_training=True \
  --reader=ImageCaptionReader \
  --multiple_references=True \
  --exclude_variable_patterns="OptimizeLoss/.+/Adam.*,OptimizeLoss/beta1_power,OptimizeLoss/beta2_power" \
  --optimizer=Adam \
  --document_frequency_file="${DOCUMENT_FREQUENCY_FILE}"

