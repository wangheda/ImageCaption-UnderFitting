#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

device=0

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/New_Ranker_data/5170e21e04710bdcf3f85cffc39bee4d0acc0e74/ranker_tfrecords"
MODEL_DIR="${DIR}/model"
image_model=InceptionV3Model
text_model=LstmModel
match_model=CosModel

model_dir_name=multi_hyp_ranker_cosine_model


cd ranker && CUDA_VISIBLE_DEVICES=$device python train.py \
  --input_file_pattern="${TFRECORD_DIR}/rankertrain-*.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --image_model=${image_model} \
  --text_model=${text_model} \
  --match_model=${match_model} \
  --lstm_cell_type="highway" \
  --num_lstm_layers=2 \
  --train_inception_with_decay=True \
  --optimizer=Adam \
  --initial_learning_rate=0.0001 \
  --learning_rate_decay_factor=0.9 \
  --num_epochs_per_decay=1.0 \
  --support_ingraph=True \
  --number_of_steps=400000 \
  --batch_size=30 \
  --num_examples_per_epoch=225000 \
  --use_multi_hyp_ranker=True
