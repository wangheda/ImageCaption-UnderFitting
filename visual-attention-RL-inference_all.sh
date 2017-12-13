#!/bin/bash

model_name="show_and_tell_advanced_model_visual_attention_rl_adam_lr5e-5_decay308"
model=ShowAndTellAdvancedModel
num_processes=1
gpu_fraction=1.0
device=0
ckpt=840942

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_DIR="${DIR}/model/${model_name}"
CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
BASE_OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.inference_all"

TRAIN_IMAGE_DIR="${DIR}/data/ai_challenger_caption_train_20170902/caption_train_images_20170902"
VALIDATE_IMAGE_DIR="${DIR}/data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
TEST_IMAGE_DIR="${DIR}/data/ai_challenger_caption_test1_20170923/caption_test1_images_20170923"

cd ${DIR}/im2txt

for mode in TRAIN VALIDATE TEST; do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${mode}"
  IMAGE_DIR_NAME="${mode}_IMAGE_DIR"
  IMAGE_DIR=${!IMAGE_DIR_NAME}

  mkdir -p $OUTPUT_DIR

  for prefix in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do 
    sleep 2s
    CUDA_VISIBLE_DEVICES=$device python inference_all.py \
      --input_file_pattern="${IMAGE_DIR}/${prefix}*.jpg" \
      --checkpoint_path=${CHECKPOINT_PATH} \
      --vocab_file=${DIR}/data/word_counts.txt \
      --output=${OUTPUT_DIR}/part-${prefix}.json \
      --model=${model} \
      --support_ingraph=True \
      --gpu_memory_fraction=$gpu_fraction \
      --inception_return_tuple=True \
      --use_attention_wrapper=True \
      --attention_mechanism=BahdanauAttention \
      --num_lstm_layers=1
  done
done

echo output saved to ${BASE_OUTPUT_DIR}

cd ${DIR}
