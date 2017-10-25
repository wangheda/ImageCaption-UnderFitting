#!/bin/bash

model_name="semantic_attention_model_join"
ckpt=164078
device=""

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_DIR="${DIR}/model/${model_name}"
IMAGE_DIR="${DIR}/data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"


cd ${DIR}/im2txt

CUDA_VISIBLE_DEVICES=$device python test_inference_semantic_attention_model.py \
    --input_file_pattern="${IMAGE_DIR}/*.jpg" \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --vocab_file=${DIR}/data/word_counts.txt \
    --concepts_file=${DIR}/data/concepts.txt \
    --model=SemanticAttentionModel \
    --use_semantic=True
