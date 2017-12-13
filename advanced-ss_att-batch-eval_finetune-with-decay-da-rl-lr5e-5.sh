#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#model_name="show_and_tell_advanced_model_attention_finetune_with_decay_da_rl"
#model_name="show_and_tell_advanced_model_visual_attention_rl_adam_lr5e-5_decay308"
model_name="show_and_tell_advanced_model_visual_attention_rl_adam_lr5e-5_decay308_bsa"
num_processes=1
gpu_fraction=1
device=1
model=ShowAndTellAdvancedModel

MODEL_DIR="${DIR}/model/${model_name}"
for ckpt in $(ls ${MODEL_DIR} | python ${DIR}/tools/every_n_step.py 20000); do 
  # the script directory
  VALIDATE_TFRECORD_FILE="${DIR}/data/Newloc_TFRecord_data/validate*.tfrecord"
  VALIDATE_REFERENCE_FILE="${DIR}/data/ai_challenger_caption_validation_20170910/reference.json"

  CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
  OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.eval"

  mkdir $OUTPUT_DIR

  cd ${DIR}/im2txt
  if [ ! -f ${OUTPUT_DIR}/out.json ]; then
    CUDA_VISIBLE_DEVICES=$device python batch_inference.py \
      --input_file_pattern="${VALIDATE_TFRECORD_FILE}" \
      --checkpoint_path=${CHECKPOINT_PATH} \
      --vocab_file=${DIR}/data/word_counts.txt \
      --output=${OUTPUT_DIR}/out.json \
      --reader=ImageCaptionTestReader \
      --batch_size=10 \
      --model=${model} \
      --inception_return_tuple=True \
      --use_attention_wrapper=True \
      --attention_mechanism=BahdanauAttention \
      --num_lstm_layers=1 \
      --support_ingraph=True \
      --gpu_memory_fraction=$gpu_fraction
    echo output saved to ${OUTPUT_DIR}/out.json
  fi

  if [ ! -f ${OUTPUT_DIR}/out.eval ]; then
    python ${DIR}/tools/eval/run_evaluations.py --submit ${OUTPUT_DIR}/out.json --ref $VALIDATE_REFERENCE_FILE | tee ${OUTPUT_DIR}/out.eval | grep ^Eval
    echo eval result saved to ${OUTPUT_DIR}/out.eval
  fi
done
