#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# image directory
VALIDATE_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
TRAIN_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_images_20170902"

# directory
base_dir="${DIR}/../data/New_Ranker_data/5170e21e04710bdcf3f85cffc39bee4d0acc0e74"

for part in VALIDATE TRAIN; do
  if [ $part == "TRAIN" ]; then
    IMAGE_DIR=$TRAIN_IMAGE_DIR
    LIST="0 1 2 3 4 5 6 7 8 9 a b c d e f"
    SHARD=10
  else
    IMAGE_DIR=$VALIDATE_IMAGE_DIR
    LIST="0 2 4 6 8 a c e"
    SHARD=1
  fi

  image_dir=$IMAGE_DIR
  output_dir=${base_dir}/ranker_tfrecords

  maxlen=30
  num_shards=$SHARD

  for c in $LIST; do
    csv_file=${base_dir}/csv-${part}-${c}
    CUDA_VISIBLE_DEVICES="" python ${DIR}/build_new_rank_tfrecords.py \
            --word_counts_input_file=${DIR}/../data/word_counts.txt \
            --input_file=$csv_file \
            --num_shards=$num_shards \
            --maxlen=$maxlen \
            --lines_per_image=15 \
            --image_dir=$image_dir \
            --output_prefix="rankertrain-${part}-${c}" \
            --output_dir=$output_dir \
  done
done





