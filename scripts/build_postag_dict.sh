#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# output directories
VOCAB_FILE="${DIR}/../data/word_counts.txt"

# input directories
TRAIN_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json"

# run the script
python ${DIR}/build_postag_dict.py \
  --output_dir=$OUTPUT_DIR \
  --word_counts_input_file=$VOCAB_FILE \
  --postags_output_file=${DIR}/../data/postags.txt \
  --word2postag_output_file=${DIR}/../data/word2postag.txt \
  --train_captions_file=$TRAIN_CAPTIONS_FILE \
  --min_word_count=$MIN_WORD_COUNT

