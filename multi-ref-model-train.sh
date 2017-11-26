#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

command="cmd:$1"

device=1
model=MultiRefModel
model_dir_name=multi_ref_model

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/Loc_TFRecord_data"
DOCUMENT_FREQUENCY_FILE="${DIR}/data/document_frequency.json"
MODEL_DIR="${DIR}/model/$model_dir_name"

[ ! -d $MODEL_DIR ] && mkdir $MODEL_DIR

cd im2txt

SUB_MODEL_DIR="$MODEL_DIR/mle_train"
STEPS=7000
if [ $command == "cmd:train" ]; then
  echo "command is train"
  CUDA_VISIBLE_DEVICES=$device python train.py \
    --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
    --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
    --model=${model} \
    --l2_normalize_image=True \
    --vocab_size=10000 \
    --train_dir="${SUB_MODEL_DIR}" \
    --initial_learning_rate=2.0 \
    --learning_rate_decay_factor=0.77 \
    --num_epochs_per_decay=1.0 \
    --inception_return_tuple=True \
    --use_scheduled_sampling=False \
    --use_attention_wrapper=True \
    --attention_mechanism=BahdanauAttention \
    --num_lstm_layers=1 \
    --number_of_steps=$STEPS \
    --reader=ImageCaptionReader \
    --multiple_references=True 
fi
PREV_SUB_MODEL_DIR=$SUB_MODEL_DIR
PREV_STEPS=$STEPS

SUB_MODEL_DIR="$MODEL_DIR/mle_finetune"
STEPS=105000
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
    --model=${model} \
    --l2_normalize_image=True \
    --vocab_size=10000 \
    --train_dir="${SUB_MODEL_DIR}" \
    --initial_learning_rate=0.8 \
    --learning_rate_decay_factor=0.77 \
    --num_epochs_per_decay=1.0 \
    --inception_return_tuple=True \
    --use_scheduled_sampling=False \
    --use_attention_wrapper=True \
    --attention_mechanism=BahdanauAttention \
    --num_lstm_layers=1 \
    --train_inception_with_decay=True \
    --number_of_steps=$STEPS \
    --swap_memory=True \
    --reader=ImageCaptionReader \
    --multiple_references=True
fi
PREV_SUB_MODEL_DIR=$SUB_MODEL_DIR
PREV_STEPS=$STEPS

SUB_MODEL_DIR="$MODEL_DIR/rl_finetune"
STEPS=140000
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
    --l2_normalize_image=True \
    --vocab_size=10000 \
    --save_interval_secs=300 \
    --keep_checkpoint_every_n_hours=0.166 \
    --initial_learning_rate=0.8 \
    --learning_rate_decay_factor=0.77 \
    --num_epochs_per_decay=1.0 \
    --inception_return_tuple=True \
    --use_scheduled_sampling=False \
    --use_attention_wrapper=True \
    --attention_mechanism=BahdanauAttention \
    --num_lstm_layers=1 \
    --number_of_steps=$STEPS \
    --rl_training=True \
    --rl_training_loss="SelfCriticalLoss" \
    --reader=ImageCaptionReader \
    --multiple_references=True \
    --document_frequency_file="${DOCUMENT_FREQUENCY_FILE}"
fi
PREV_SUB_MODEL_DIR=$SUB_MODEL_DIR
PREV_STEPS=$STEPS

if [ $command == "cmd:all" ]; then
  bash -x ${DIR}/multi-ref-model-train.sh train
  bash -x ${DIR}/multi-ref-model-train.sh finetune
  bash -x ${DIR}/multi-ref-model-train.sh rl_finetune
fi
