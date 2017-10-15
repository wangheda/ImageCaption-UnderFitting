#!/bin/bash

# parameters to set
MIN_WORD_COUNT=4

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# output directories
OUTPUT_DIR="${DIR}/../data/Aug_TFRecord_data"
VOCAB_FILE="${DIR}/../data/word_counts.txt"

# input directories
TRAIN_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_images_20170902"
TRAIN_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json"
VALIDATE_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
VALIDATE_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json"

# empty the output dir
rm ${OUTPUT_DIR}/*.tfrecord

# run the script
CUDA_VISIBLE_DEVICES=1 python ${DIR}/build_tfrecords.py \
  --output_dir=$OUTPUT_DIR \
  --word_counts_input_file=$VOCAB_FILE \
  --train_image_dir=$TRAIN_IMAGE_DIR \
  --validate_image_dir=$VALIDATE_IMAGE_DIR \
  --train_captions_file=$TRAIN_CAPTIONS_FILE \
  --validate_captions_file=$VALIDATE_CAPTIONS_FILE \
  --build_flip_caption=True \
  --min_word_count=$MIN_WORD_COUNT
