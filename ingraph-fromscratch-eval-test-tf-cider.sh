#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

model_name="show_and_tell_in_graph_model_fromscratch"
num_processes=1
device=1
model=ShowAndTellInGraphModel

MODEL_DIR="${DIR}/model/${model_name}"
for ckpt in $(ls ${MODEL_DIR} | python ${DIR}/tools/every_n_step.py 20000 | tail -n 15 | tac); do 
  # the script directory
  VALIDATE_IMAGE_DIR="${DIR}/data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
  VALIDATE_REFERENCE_FILE="${DIR}/data/ai_challenger_caption_validation_20170910/reference.json"

  CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
  OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.eval"

  mkdir $OUTPUT_DIR

  cd ${DIR}/im2txt

  if [ ! -f ${OUTPUT_DIR}/out.json ]; then
    CUDA_VISIBLE_DEVICES=$device python inference.py \
      --input_file_pattern="${VALIDATE_IMAGE_DIR}/*.jpg" \
      --checkpoint_path=${CHECKPOINT_PATH} \
      --vocab_file=${DIR}/data/word_counts.txt \
      --output=${OUTPUT_DIR}/out.json \
      --model=${model} \
      --support_ingraph=True
    echo output saved to ${OUTPUT_DIR}/out.json
  fi

  if [ ! -f ${OUTPUT_DIR}/out.eval_with_tf_cider ]; then
    python ${DIR}/tools/eval/run_evaluations_with_tf_cider.py \
        --vocab_file=${DIR}/data/word_counts.txt \
        --submit ${OUTPUT_DIR}/out.json \
        --ref $VALIDATE_REFERENCE_FILE | tee ${OUTPUT_DIR}/out.eval_with_tf_cider | grep ^Eval
    echo eval result saved to ${OUTPUT_DIR}/out.eval_with_tf_cider
  fi
  exit 1
done

