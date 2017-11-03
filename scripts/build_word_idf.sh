
#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# output directories
VOCAB_FILE="${DIR}/../data/word_counts.txt"

# input directories
TRAIN_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json"

# run the script
python ${DIR}/build_word_idf.py \
  --train_captions_file=$TRAIN_CAPTIONS_FILE \
  --word_counts_input_file=$VOCAB_FILE \
  --word_idf_output_file=${DIR}/../data/word_idf.txt
