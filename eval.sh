#!/bin/bash

model_name="show_and_tell_model"
num_processes=3
gpu_fraction=0.28
device=1
ckpt=101463

for ckpt in 65271 129123 193009 257007 322481; do 
  # the script directory
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  MODEL_DIR="${DIR}/model/${model_name}"
  IMAGE_DIR="${DIR}/data/ai_challenger_caption_test1_20170923/caption_test1_images_20170923"
  VALIDATE_REFERENCE_FILE="${DIR}/data/ai_challenger_caption_validation_20170910/reference.json"

  CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
  OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.eval"

  mkdir $OUTPUT_DIR

  cd ${DIR}/im2txt

  for prefix in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do 
    echo "CUDA_VISIBLE_DEVICES=$device python inference.py \
      --input_file_pattern='${IMAGE_DIR}/${prefix}*.jpg' \
      --checkpoint_path=${CHECKPOINT_PATH} \
      --vocab_file=${DIR}/data/word_counts.txt \
      --output=${OUTPUT_DIR}/part-${prefix}.json \
      --gpu_memory_fraction=$gpu_fraction"
  done | parallel -j $num_processes

  cd ${DIR}

  python tools/merge_json_lists.py ${OUTPUT_DIR}/part-?.json > ${OUTPUT_DIR}/out.json

  echo output saved to ${OUTPUT_DIR}/out.json

  cd ${DIR}/tools/eval

  python run_evaluations.py --submit ${OUTPUT_DIR}/out.json --ref $VALIDATE_REFERENCE_FILE | tee ${OUTPUT_DIR}/out.eval | grep ^Eval

  echo eval result saved to ${OUTPUT_DIR}/out.eval
done
