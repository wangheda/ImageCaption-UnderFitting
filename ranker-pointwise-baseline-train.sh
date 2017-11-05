#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/Ranker_TFRecord_data/5170e21e04710bdcf3f85cffc39bee4d0acc0e74"
MODEL_DIR="${DIR}/model"
image_model=InceptionV3Model
text_model=LstmModel
match_model=MlpModel

model_dir_name=ranker_pointwise_baseline_model

cd ranker && CUDA_VISIBLE_DEVICES=0 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/rankertrain-*.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --image_model=${image_model} \
  --text_model=${text_model} \
  --match_model=${match_model} \
  --loss="CrossEntropyLoss" \
  --mlp_num_layers=2 \
  --mlp_num_units="512,256" \
  --mlp_type_activation="sigmoid" \
  --lstm_cell_type="highway" \
  --num_lstm_layers=2 \
  --train_inception_with_decay=True \
  --optimizer=Adam \
  --initial_learning_rate=0.001 \
  --learning_rate_decay_factor=0.9 \
  --num_epochs_per_decay=1.0 \
  --support_ingraph=True \
  --number_of_steps=400000
