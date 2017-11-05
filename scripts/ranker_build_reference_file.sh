#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# input
TRAIN_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json"
# output
TRAIN_REFERENCE_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/reference.json"

# input
VALIDATE_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json"
# output
VALIDATE_REFERENCE_FILE="${DIR}/../data/ai_challenger_caption_validation_20170910/reference.json"

if [ ! -f $VALIDATE_REFERENCE_FILE ]; then
  # run the script
  python ${DIR}/build_reference_file.py \
    --captions_file=$VALIDATE_CAPTIONS_FILE \
    --output_file=$VALIDATE_REFERENCE_FILE 
else
  echo "Skip generating $VALIDATE_REFERENCE_FILE because it already exists, delete it manually and rerun if you want it re-generated."
fi

if [ ! -f $TRAIN_REFERENCE_FILE ]; then
  # run the script
  python ${DIR}/build_reference_file.py \
    --captions_file=$TRAIN_CAPTIONS_FILE \
    --output_file=$TRAIN_REFERENCE_FILE 
else
  echo "Skip generating $TRAIN_REFERENCE_FILE because it already exists, delete it manually and rerun if you want it re-generated."
fi

