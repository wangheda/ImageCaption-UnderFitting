#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

command="cmd:$1"

device=0
model=ShowAndTellAdvancedModel
model_dir_name=localization_attention_model

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/Newloc_TFRecord_data"
DOCUMENT_FREQUENCY_FILE="${DIR}/data/document_frequency.json"
MODEL_DIR="${DIR}/model/$model_dir_name"

[ ! -d $MODEL_DIR ] && mkdir $MODEL_DIR

cd im2txt

SUB_MODEL_DIR="$MODEL_DIR/mle_train"
STEPS=35000
if [ $command == "cmd:train" ]; then
  echo "command is train"
  CUDA_VISIBLE_DEVICES=$device python train.py \
    --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
    --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
    --train_dir="${SUB_MODEL_DIR}" \
    --model=${model} \
    --localization_attention=True \
    --initial_learning_rate=1.0 \
    --learning_rate_decay_factor=0.66 \
    --inception_return_tuple=True \
    --use_scheduled_sampling=False \
    --use_attention_wrapper=True \
    --attention_mechanism=BahdanauAttention \
    --num_lstm_layers=1 \
    --support_ingraph=True \
    --reader=ImageCaptionReader \
    --cropping_images=False \
    --number_of_steps=$STEPS
fi
PREV_SUB_MODEL_DIR=$SUB_MODEL_DIR
PREV_STEPS=$STEPS

SUB_MODEL_DIR="$MODEL_DIR/mle_finetune"
STEPS=525000
if [ $command == "cmd:finetune" ]; then
  echo "command is finetune"
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
    --localization_attention=True \
    --initial_learning_rate=1.0 \
    --learning_rate_decay_factor=0.66 \
    --inception_return_tuple=True \
    --use_scheduled_sampling=False \
    --use_attention_wrapper=True \
    --attention_mechanism=BahdanauAttention \
    --num_lstm_layers=1 \
    --support_ingraph=True \
    --train_inception_with_decay=True \
    --swap_memory=True \
    --reader=ImageCaptionReader \
    --cropping_images=False \
    --number_of_steps=$STEPS
fi
PREV_SUB_MODEL_DIR=$SUB_MODEL_DIR
PREV_STEPS=$STEPS

SUB_MODEL_DIR="$MODEL_DIR/rl_finetune"
STEPS=595000
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
    --localization_attention=True \
    --initial_learning_rate=1.0 \
    --learning_rate_decay_factor=0.66 \
    --inception_return_tuple=True \
    --use_scheduled_sampling=False \
    --use_attention_wrapper=True \
    --attention_mechanism=BahdanauAttention \
    --num_lstm_layers=1 \
    --support_ingraph=True \
    --reader=ImageCaptionReader \
    --multiple_references=True \
    --cropping_images=False \
    --rl_training=True \
    --rl_training_loss="SelfCriticalLoss" \
    --document_frequency_file="${DOCUMENT_FREQUENCY_FILE}" \
    --number_of_steps=$STEPS
fi
PREV_SUB_MODEL_DIR=$SUB_MODEL_DIR
PREV_STEPS=$STEPS

if [ $command == "cmd:all" ]; then
  bash -x ${DIR}/$0 train
  bash -x ${DIR}/$0 finetune
  bash -x ${DIR}/$0 rl_finetune
fi
