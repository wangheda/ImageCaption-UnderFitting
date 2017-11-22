
#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# output directories
VOCAB_FILE="${DIR}/../data/word_counts.txt"

# input directories
TRAIN_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json"

# run the script
python ${DIR}/build_document_frequency.py \
    --annotation_file=$TRAIN_CAPTIONS_FILE \
    --vocab_file=$VOCAB_FILE \
    --output_file=${DIR}/../data/document_frequency.json
