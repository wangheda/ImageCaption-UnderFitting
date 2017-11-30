#!/bin/bash

device=1

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/Ranker_TFRecord_data_v2/"
MODEL_DIR="${DIR}/model"
image_model=InceptionV3Model
text_model=LstmModel
match_model=CosModel

model_dir_name=ranker_v2_pairwise_baseline_sigmoid_model

cd ranker && CUDA_VISIBLE_DEVICES=${device} python train.py \
  --input_file_pattern="${TFRECORD_DIR}/rankertrain-*.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --lines_per_image=10 \
  --image_model=${image_model} \
  --text_model=${text_model} \
  --match_model=${match_model} \
  --loss="HingeLoss" \
  --hinge_loss_margin=0.2 \
  --lstm_cell_type="residual" \
  --num_lstm_layers=2 \
  --cos_type_activation="sigmoid" \
  --train_inception_with_decay=True \
  --optimizer=Adam \
  --initial_learning_rate=0.001 \
  --learning_rate_decay_factor=0.9 \
  --num_epochs_per_decay=2.0 \
  --support_ingraph=True \
  --number_of_steps=100000
