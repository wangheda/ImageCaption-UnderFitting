#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/Aug_TFRecord_data"
MODEL_DIR="${DIR}/model"

model=ShowAndTellAdvancedModel
model_dir_name=show_and_tell_advanced_model_visual_attention_highway_4layers

cd im2txt

CUDA_VISIBLE_DEVICES=1 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
  #--inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --model=${model} \
  --batch_size=32 \
  --initial_learning_rate=1.25 \
  --learning_rate_decay_factor=0.9 \
  --num_epochs_per_decay=2.0 \
  --inception_return_tuple=True \
  --use_scheduled_sampling=False \
  --use_attention_wrapper=True \
  --support_flip=True \
  --attention_mechanism=BahdanauAttention \
  --num_lstm_layers=4 \
  --lstm_cell_type="highway" \
  --num_lstm_units=512 \
  --num_attention_depth=256 \
  --embedding_size=256 \
  --support_ingraph=True \
  --number_of_steps=30000

CUDA_VISIBLE_DEVICES=1 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --model=${model} \
  --batch_size=32 \
  --initial_learning_rate=1.25 \
  --learning_rate_decay_factor=0.9 \
  --num_epochs_per_decay=2.0 \
  --inception_return_tuple=True \
  --use_scheduled_sampling=False \
  --use_attention_wrapper=True \
  --support_flip=True \
  --attention_mechanism=BahdanauAttention \
  --num_lstm_layers=4 \
  --lstm_cell_type="highway" \
  --num_lstm_units=512 \
  --num_attention_depth=256 \
  --embedding_size=256 \
  --support_ingraph=True \
  --train_inception_with_decay=True \
  --number_of_steps=600000

# note: num_lstm_units = 512 + 32 + 128 + 128
#                     word_embedding + postag + char + visual
