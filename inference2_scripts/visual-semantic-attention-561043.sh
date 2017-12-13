#!/bin/bash

model_name="show_and_tell_advanced_model_new_vis_sem_attention"
model=ShowAndTellAdvancedModel
num_processes=1
gpu_fraction=1.0
device=1
ckpt=561043

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_DIR="${DIR}/../model/${model_name}"

TFRECORD_FILE="${DIR}/../data/Newloc_TFRecord_data/test2*.tfrecord"

VOCAB_FILE="${DIR}/../data/word_counts.txt"
CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.test2"

mkdir -p $OUTPUT_DIR

cd ${DIR}/../im2txt

for i in {1..20}; do 
  if [ ! -f ${OUTPUT_DIR}/run-${i}.json ]; then
    CUDA_VISIBLE_DEVICES=$device python batch_inference.py \
      --input_file_pattern="$TFRECORD_FILE" \
      --checkpoint_path=${CHECKPOINT_PATH} \
      --vocab_file=$VOCAB_FILE \
      --output=${OUTPUT_DIR}/run-${i}.json \
      --model=${model} \
      --reader=ImageCaptionTestReader \
      --batch_size=30 \
      --fuzzy_test=True \
      --inception_return_tuple=True \
      --use_attention_wrapper=True \
      --attention_mechanism=BahdanauAttention \
      --num_lstm_layers=1 \
      --predict_words_via_image_output=True \
      --use_semantic_attention=True \
      --semantic_attention_type="topk" \
      --semantic_attention_topk_word=10 \
      --use_separate_embedding_for_semantic_attention=True \
      --semantic_attention_word_hash_depth=128 \
      --support_ingraph=True
    echo output saved to ${OUTPUT_DIR}/run-${i}.json
  fi
done

python ${DIR}/../tools/captions_vote.py ${OUTPUT_DIR}/run-*.json > ${OUTPUT_DIR}/out.json

echo $OUTPUT_DIR >> test2_location.list
