#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INCEPTION_CHECKPOINT="${DIR}/pretrained_model/inception_v3/inception_v3.ckpt"
TFRECORD_DIR="${DIR}/data/TFRecord_data"
MODEL_DIR="${DIR}/model"
model=ShowAndTellAdvancedModel

model_dir_name=show_and_tell_advanced_model_topk_semantic_attention_2lexical

cd im2txt && CUDA_VISIBLE_DEVICES=0 python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-?????.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/${model_dir_name}" \
  --model=${model} \
  --initial_learning_rate=1.0 \
  --learning_rate_decay_factor=0.6 \
  --train_inception_with_decay=True \
  --use_scheduled_sampling=False \
  --attention_mechanism=BahdanauAttention \
  --num_lstm_layers=1 \
  --predict_words_via_image_output=True \
  --use_semantic_attention=True \
  --use_separate_embedding_for_semantic_attention=True \
  --semantic_attention_type="topk" \
  --use_lexical_embedding=True \
  --lexical_mapping_file="${DIR}/data/word2postag.txt,${DIR}/data/word2char.txt" \
  --lexical_embedding_type="postag,char" \
  --embedding_size=256 \
  --lexical_embedding_size="32,64" \
  --support_ingraph=True \
  --number_of_steps=105000
