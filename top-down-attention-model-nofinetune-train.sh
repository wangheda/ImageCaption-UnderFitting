#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

command="cmd:$1"

device=0
model=TopDownAttentionModel
model_dir_name=top_down_attention_model

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/Newloc_TFRecord_data"
DOCUMENT_FREQUENCY_FILE="${DIR}/data/document_frequency.json"
MODEL_DIR="${DIR}/model/$model_dir_name"

[ ! -d $MODEL_DIR ] && mkdir $MODEL_DIR

cd im2txt

SUB_MODEL_DIR="$MODEL_DIR/mle_train"
STEPS=400000
if [ $command == "cmd:train" ]; then
  echo "command is train"
  CUDA_VISIBLE_DEVICES=$device python train.py \
    --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
    --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
    --train_dir="${SUB_MODEL_DIR}" \
    --model=${model} \
    --batch_size=42 \
    --optimizer=Momentum \
    --initial_learning_rate=0.1 \
    --learning_rate_decay_factor=0.9 \
    --num_epochs_per_decay=5.0 \
    --inception_return_tuple=True \
    --use_scheduled_sampling=False \
    --support_ingraph=True \
    --num_lstm_units=512 \
    --num_attention_depth=512 \
    --embedding_size=512 \
    --swap_memory=True \
    --reader=ImageCaptionReader \
    --localization_attention=True \
    --cropping_images=False \
    --number_of_steps=$STEPS
fi
PREV_SUB_MODEL_DIR=$SUB_MODEL_DIR
PREV_STEPS=$STEPS

SUB_MODEL_DIR="$MODEL_DIR/rl_finetune"
STEPS=500000
if [ $command == "cmd:rl_finetune" ]; then
  echo "command is rl_finetune"
  if [ ! -d $SUB_MODEL_DIR ]; then
    mkdir -p $SUB_MODEL_DIR
    cp ${PREV_SUB_MODEL_DIR}/model.ckpt-${PREV_STEPS}.* ${SUB_MODEL_DIR}/
    echo "model_checkpoint_path: \"${PREV_SUB_MODEL_DIR}/model.ckpt-${PREV_STEPS}\"" > ${SUB_MODEL_DIR}/checkpoint
  fi

  CUDA_VISIBLE_DEVICES=$device python train.py \
    --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
    --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
    --train_dir="${SUB_MODEL_DIR}" \
    --model=${model} \
    --batch_size=42 \
    --optimizer=Momentum \
    --initial_learning_rate=0.1 \
    --learning_rate_decay_factor=0.9 \
    --num_epochs_per_decay=5.0 \
    --inception_return_tuple=True \
    --support_ingraph=True \
    --num_lstm_units=512 \
    --num_attention_depth=512 \
    --embedding_size=512 \
    --reader=ImageCaptionReader \
    --localization_attention=True \
    --cropping_images=False \
    --multiple_references=True \
    --rl_training=True \
    --rl_training_loss="SelfCriticalLoss" \
    --document_frequency_file="${DOCUMENT_FREQUENCY_FILE}" \
    --exclude_variable_patterns='OptimizeLoss/InceptionV3/.*' \
    --number_of_steps=$STEPS
fi
PREV_SUB_MODEL_DIR=$SUB_MODEL_DIR
PREV_STEPS=$STEPS

if [ $command == "cmd:all" ]; then
  bash -x ${DIR}/$0 train
  bash -x ${DIR}/$0 rl_finetune
fi
