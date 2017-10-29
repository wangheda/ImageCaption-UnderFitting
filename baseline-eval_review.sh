#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DATA="/media/diskb/x39_share/ImageCaption"
model_name="review_model"
num_processes=3
gpu_fraction=0.9
device=1
ckpt=298918

for ckpt in 298918; do 
  # the script directory
  MODEL_DIR="${DIR}/model/${model_name}"
  VALIDATE_IMAGE_DIR="${DATA}/data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
  VALIDATE_REFERENCE_FILE="/home/suying/ImageCaption/ImageCaption-xqx/model/reference.json"

  CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
  OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.eval"

  mkdir $OUTPUT_DIR

  cd ${DIR}/im2txt

  for prefix in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do 
    if [ ! -f ${OUTPUT_DIR}/part-${prefix}.json ]; then
      echo "CUDA_VISIBLE_DEVICES=$device python inference.py \
        --input_file_pattern='${VALIDATE_IMAGE_DIR}/${prefix}*.jpg' \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --vocab_file=${DATA}/data/word_counts.txt \
        --output=${OUTPUT_DIR}/part-${prefix}.json \
        --gpu_memory_fraction=$gpu_fraction"
      CUDA_VISIBLE_DEVICES=$device python inference.py \
        --input_file_pattern="${VALIDATE_IMAGE_DIR}/${prefix}*.jpg" \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --vocab_file=${DATA}/data/word_counts.txt \
        --output=${OUTPUT_DIR}/part-${prefix}.json \
        --model=ReviewnetworkModel \
        --use_attention=True \
        --gpu_memory_fraction=$gpu_fraction
    fi
  done

  python ${DIR}/tools/merge_json_lists.py ${OUTPUT_DIR}/part-?.json > ${OUTPUT_DIR}/out.json
  echo output saved to ${OUTPUT_DIR}/out.json

  python ${DIR}/tools/eval/run_evaluations.py --submit ${OUTPUT_DIR}/out.json --ref $VALIDATE_REFERENCE_FILE | tee ${OUTPUT_DIR}/out.eval | grep ^Eval
  echo eval result saved to ${OUTPUT_DIR}/out.eval
done
