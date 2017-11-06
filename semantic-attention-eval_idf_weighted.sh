#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

model_name="semantic_attention_model_join_idf_weighted"
num_processes=2
gpu_fraction=0.45
device=1
model=SemanticAttentionModel

MODEL_DIR="${DIR}/model/${model_name}"
for ckpt in $(ls ${MODEL_DIR} | python ${DIR}/tools/every_n_step.py 20000); do 
  # the script directory
  VALIDATE_IMAGE_DIR="${DIR}/data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
  VALIDATE_REFERENCE_FILE="${DIR}/data/ai_challenger_caption_validation_20170910/reference.json"

  CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
  OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.eval"

  mkdir $OUTPUT_DIR

  cd ${DIR}/im2txt

  for prefix in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do 
    if [ ! -f ${OUTPUT_DIR}/part-${prefix}.json ]; then
      echo "CUDA_VISIBLE_DEVICES=$device python inference.py \
        --input_file_pattern='${VALIDATE_IMAGE_DIR}/${prefix}*.jpg' \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --vocab_file=${DIR}/data/word_counts.txt \
        --attributes_file=${DIR}/data/attributes.txt \
        --output=${OUTPUT_DIR}/part-${prefix}.json \
        --model=${model} \
        --support_ingraph=True \
        --gpu_memory_fraction=$gpu_fraction"
    fi
  done | parallel -j $num_processes

  if [ ! -f ${OUTPUT_DIR}/out.json ]; then
    python ${DIR}/tools/merge_json_lists.py ${OUTPUT_DIR}/part-?.json > ${OUTPUT_DIR}/out.json
    echo output saved to ${OUTPUT_DIR}/out.json
  fi

  if [ ! -f ${OUTPUT_DIR}/out.eval ]; then
    python ${DIR}/tools/eval/run_evaluations.py --submit ${OUTPUT_DIR}/out.json --ref $VALIDATE_REFERENCE_FILE | tee ${OUTPUT_DIR}/out.eval | grep ^Eval
    echo eval result saved to ${OUTPUT_DIR}/out.eval
  fi
done
