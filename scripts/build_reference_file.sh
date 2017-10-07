#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# input directories
VALIDATE_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json"

# output directories
VALIDATE_REFERENCE_FILE="${DIR}/../data/ai_challenger_caption_validation_20170910/reference.json"

# run the script
python ${DIR}/build_reference_file.py \
  --captions_file=$VALIDATE_CAPTIONS_FILE \
  --output_file=$VALIDATE_REFERENCE_FILE 
