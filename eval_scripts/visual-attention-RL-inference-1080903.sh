#!/bin/bash

model_name="show_and_tell_advanced_model_visual_attention_rl_adam_lr5e-5_decay308"
model=ShowAndTellAdvancedModel
num_processes=1
gpu_fraction=1.0
device=1
ckpt=1080903

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_DIR="${DIR}/../model/${model_name}"

TFRECORD_FILE="${DIR}/../data/Newloc_TFRecord_data/validate*.tfrecord"
VALIDATE_REFERENCE_FILE="${DIR}/data/ai_challenger_caption_validation_20170910/reference.json"

VOCAB_FILE="${DIR}/../data/word_counts.txt"
CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-$ckpt"
OUTPUT_DIR="${MODEL_DIR}/model.ckpt-${ckpt}.eval"

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
      --support_ingraph=True
    echo output saved to ${OUTPUT_DIR}/run-${i}.json
  fi
done

python ${DIR}/../tools/captions_vote.py ${OUTPUT_DIR}/run-*.json > ${OUTPUT_DIR}/out.json

if [ ! -f ${OUTPUT_DIR}/out.eval ]; then
  python ${DIR}/../tools/eval/run_evaluations.py --submit ${OUTPUT_DIR}/out.json --ref $VALIDATE_REFERENCE_FILE | tee ${OUTPUT_DIR}/out.eval | grep ^Eval
  echo eval result saved to ${OUTPUT_DIR}/out.eval
fi
