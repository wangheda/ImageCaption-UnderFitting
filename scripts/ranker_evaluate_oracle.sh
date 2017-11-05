#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# input
INFERENCE_ALL="${DIR}/../resources/inference_all.list"

# caption
VALIDATE_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json"
TRAIN_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json"

# image directory
VALIDATE_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
TRAIN_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_images_20170902"

# directory
base_dir="${DIR}/../model/combination"
hash_dir1=$(sha1sum $INFERENCE_ALL | cut -d' ' -f 1)

# reference file
ref_dir=${base_dir}/reference
[ ! -d $ref_dir ] && exit 1
[ ! -d ${ref_dir}/VALIDATE ] && exit 1

for part in VALIDATE; do
  for c in 13579bdf; do
    CAPTIONS_FILE=$VALIDATE_CAPTIONS_FILE
    ref_file=${ref_dir}/VALIDATE/reference-${c}.json
    if [ ! -f $ref_file ]; then
      python ${DIR}/ranker_build_reference_file.py \
        --captions_file=$CAPTIONS_FILE \
        --prefix="1,3,5,7,9,b,d,f" \
        --output_file=$ref_file
    fi

    ranker_validate_csv_file=${base_dir}/${hash_dir1}/ranker_temp_file_csv-${part}-${c}
    rm $ranker_validate_csv_file
    for i in 1 3 5 7 9 b d f; do
      csv_file=${base_dir}/${hash_dir1}/csv-${part}-${i}
      [ ! -f $csv_file ] && exit 1
      cat $csv_file >> $ranker_validate_csv_file
    done

    ranker_validate_oracle_file=${base_dir}/${hash_dir1}/ranker_temp_file_oracle-${part}-${c}
    if [ ! -f $ranker_validate_oracle_file ]; then
      cat $ranker_validate_csv_file | python ${DIR}/ranker_convert_csv_to_oracle_prediction.py > $ranker_validate_oracle_file
    fi

    python ${DIR}/../tools/eval/run_evaluations.py --submit $ranker_validate_oracle_file --ref $ref_file 

  done
done

