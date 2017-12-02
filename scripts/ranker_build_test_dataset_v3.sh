#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# input
INFERENCE_ALL="${DIR}/../resources/inference_all_v3.list.test"

# caption
TRAIN_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json"
VALIDATE_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json"

# image directory
VALIDATE_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
TEST_IMAGE_DIR="${DIR}/../data/ai_challenger_caption_test1_20170923/caption_test1_images_20170923"

ALL_REFS_FILE="${DIR}/../data/segmented_all_refs.txt"

# directory
base_dir="${DIR}/../model/combination"
hash_dir1=$(sha1sum $INFERENCE_ALL | cut -d' ' -f 1)

# reference file
ref_dir=${base_dir}/reference
[ ! -d $ref_dir ] && mkdir -p $ref_dir
[ ! -d ${ref_dir}/VALIDATE ] && mkdir -p ${ref_dir}/VALIDATE

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
  done
done


for part in VALIDATE TEST; do
  if [ $part == "TEST" ]; then
    IMAGE_DIR=$TEST_IMAGE_DIR
    LIST="0 1 2 3 4 5 6 7 8 9 a b c d e f"
  else
    IMAGE_DIR=$VALIDATE_IMAGE_DIR
    LIST="1 3 5 7 9 b d f"
  fi

  for c in $LIST; do
    csv_filelist=${base_dir}/${hash_dir1}/filelist-${part}-${c}.test_v3
    [ -f $csv_filelist ] && rm $csv_filelist

    for model in $(cat $INFERENCE_ALL); do 
      origin_dir=${DIR}/../${model}
      hash_dir2=$(echo $model | sha1sum |  cut -d' ' -f 1)
      tmp_dir=${base_dir}/${hash_dir1}/${hash_dir2}
      if [ -d $origin_dir ]; then
        [ ! -d $tmp_dir ] && mkdir -p $tmp_dir
        [ ! -f ${tmp_dir}/model ] && echo $model > ${tmp_dir}/model

        input_file="${origin_dir}/${part}/part-${c}.json"
        output_file="${tmp_dir}/${part}/part-${c}.csv"

        [ ! -d ${tmp_dir}/${part} ] && mkdir ${tmp_dir}/${part}

        if [ ! -f $output_file ]; then
          python ${DIR}/ranker_convert_inference_file_to_csv.py \
                --input=$input_file \
                --output=$output_file 
        fi

        echo $output_file >> $csv_filelist
      fi
    done

    csv_file=${base_dir}/${hash_dir1}/csv-${part}-${c}.test_v3
    [ ! -f $csv_file ] && cat $(cat $csv_filelist) | sort > $csv_file

    output_dir=${DIR}/../data/Ranker_TFRecord_data_v3/${hash_dir1}
    image_dir=$IMAGE_DIR

    marker_file=${base_dir}/${hash_dir1}/marker-${part}-${c}.test_v3
    maxlen=30
    model_count=$(wc -l $csv_filelist | cut -d' ' -f 1)
    num_candidates=3
    lines_per_image=$(($model_count * $num_candidates))

    if [ ! -f $marker_file ]; then
      CUDA_VISIBLE_DEVICES="" python ${DIR}/build_image_caption_tfrecords_mert.py \
              --annotation_file=$TRAIN_CAPTIONS_FILE \
              --all_refs_file=$ALL_REFS_FILE \
              --word_counts_input_file=${DIR}/../data/word_counts.txt \
              --input_file=$csv_file \
              --maxlen=$maxlen \
              --image_dir=$image_dir \
              --output_prefix="rankertest-${part}-${c}" \
              --lines_per_image=$lines_per_image \
              --output_dir=$output_dir \
      && touch $marker_file
    fi

  done

done

