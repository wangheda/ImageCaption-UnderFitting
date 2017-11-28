#!/bin/bash

# parameters to set
MIN_WORD_COUNT=4

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# output directories
OUTPUT_DIR="${DIR}/../data/Newloc_TFRecord_data"
VOCAB_FILE="${DIR}/../data/word_counts.txt"

# input directories
TRAIN_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_images_20170902"
TRAIN_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json"
TRAIN_LOCALIZATIONS_FILE="${DIR}/../data/bottom_up_attention/aichallenger_train.tsv.small"

VALIDATE_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
VALIDATE_LOCALIZATIONS_FILE="${DIR}/../data/bottom_up_attention/aichallenger_validate.tsv.small"

TEST1_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_test1_20170923/caption_test1_images_20170923"
TEST1_LOCALIZATIONS_FILE="${DIR}/../data/bottom_up_attention/aichallenger_test1.tsv.small"

if [ ! -f $OUTPUT_DIR/train-00000-of-00280.tfrecord ]; then
  # run the script
  CUDA_VISIBLE_DEVICES=1 python ${DIR}/build_localization_tfrecords.py \
    --output_dir=$OUTPUT_DIR \
    --word_counts_input_file=$VOCAB_FILE \
    --train_image_dir=$TRAIN_IMAGE_DIR \
    --train_captions_file=$TRAIN_CAPTIONS_FILE \
    --train_localizations_file=$TRAIN_LOCALIZATIONS_FILE \
    --validate_image_dir=$VALIDATE_IMAGE_DIR \
    --validate_localizations_file=$VALIDATE_LOCALIZATIONS_FILE \
    --test1_image_dir=$TEST1_IMAGE_DIR \
    --test1_localizations_file=$TEST1_LOCALIZATIONS_FILE \
    --build_flip_caption=True \
    --task=train \
    --min_word_count=$MIN_WORD_COUNT
fi

if [ ! -f $OUTPUT_DIR/validate-00000-of-00008.tfrecord ]; then
  # run the script
  CUDA_VISIBLE_DEVICES=1 python ${DIR}/build_localization_tfrecords.py \
    --output_dir=$OUTPUT_DIR \
    --word_counts_input_file=$VOCAB_FILE \
    --train_image_dir=$TRAIN_IMAGE_DIR \
    --train_captions_file=$TRAIN_CAPTIONS_FILE \
    --train_localizations_file=$TRAIN_LOCALIZATIONS_FILE \
    --validate_image_dir=$VALIDATE_IMAGE_DIR \
    --validate_localizations_file=$VALIDATE_LOCALIZATIONS_FILE \
    --test1_image_dir=$TEST1_IMAGE_DIR \
    --test1_localizations_file=$TEST1_LOCALIZATIONS_FILE \
    --build_flip_caption=True \
    --task=validate \
    --min_word_count=$MIN_WORD_COUNT
fi

if [ ! -f $OUTPUT_DIR/test1-00000-of-00008.tfrecord ]; then
  # run the script
  CUDA_VISIBLE_DEVICES=1 python ${DIR}/build_localization_tfrecords.py \
    --output_dir=$OUTPUT_DIR \
    --word_counts_input_file=$VOCAB_FILE \
    --train_image_dir=$TRAIN_IMAGE_DIR \
    --train_captions_file=$TRAIN_CAPTIONS_FILE \
    --train_localizations_file=$TRAIN_LOCALIZATIONS_FILE \
    --validate_image_dir=$VALIDATE_IMAGE_DIR \
    --validate_localizations_file=$VALIDATE_LOCALIZATIONS_FILE \
    --test1_image_dir=$TEST1_IMAGE_DIR \
    --test1_localizations_file=$TEST1_LOCALIZATIONS_FILE \
    --build_flip_caption=True \
    --task=test1 \
    --min_word_count=$MIN_WORD_COUNT
fi

