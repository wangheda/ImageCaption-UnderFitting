#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# input
INFERENCE_ALL="${DIR}/../resources/inference_all.list"

# caption
VALIDATE_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json"
TRAIN_CAPTIONS_FILE="${DIR}/../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json"

# directory
base_dir="${DIR}/../model/combination"
hash_dir1=$(sha1sum $INFERENCE_ALL | cut -d' ' -f 1)

# reference file
ref_dir=${base_dir}/reference
[ ! -d $ref_dir ] && mkdir -p $ref_dir
[ ! -d ${ref_dir}/TRAIN ] && mkdir -p ${ref_dir}/TRAIN
[ ! -d ${ref_dir}/VALIDATE ] && mkdir -p ${ref_dir}/VALIDATE

for c in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  ref_file=${ref_dir}/TRAIN/reference-${c}.json
  if [ ! -f $ref_file ]; then
    python ${DIR}/ranker_build_reference_file.py \
      --captions_file=$TRAIN_CAPTIONS_FILE \
      --prefix=${c} \
      --output_file=$ref_file
  fi
done

for c in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  ref_file=${ref_dir}/VALIDATE/reference-${c}.json
  if [ ! -f $ref_file ]; then
    python ${DIR}/ranker_build_reference_file.py \
      --captions_file=$VALIDATE_CAPTIONS_FILE \
      --prefix=${c} \
      --output_file=$ref_file
  fi
done

for part in TRAIN VALIDATE; do
  for c in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
    csv_filelist=${base_dir}/${hash_dir1}/filelist-${part}-${c}
    [ -f $csv_filelist ] && rm $csv_filelist

    for model in $(cat $INFERENCE_ALL); do 
      origin_dir=${DIR}/../${model}
      hash_dir2=$(echo $model | sha1sum |  cut -d' ' -f 1)
      tmp_dir=${base_dir}/${hash_dir1}/${hash_dir2}
      if [ -d $origin_dir ]; then
        [ ! -d $tmp_dir ] && mkdir -p $tmp_dir
        [ ! -f ${tmp_dir}/model ] && echo $model > ${tmp_dir}/model

        for candidate in 1 2 3; do
          input_file="${origin_dir}/${part}/part-${c}.json"
          output_file="${tmp_dir}/${part}/part-${c}.json.no-${candidate}"

          [ ! -d ${tmp_dir}/${part} ] && mkdir ${tmp_dir}/${part}

          if [ ! -f $output_file ]; then
            python ${DIR}/ranker_split_inference_file.py \
                  --input=$input_file \
                  --output=$output_file \
                  --candidate_id=$candidate
          fi

          result_json_file=$output_file
          reference_json_file=${ref_dir}/${part}/reference-${c}.json
          output_csv_file="${tmp_dir}/${part}/part-${c}.json.no-${candidate}.csv"

          if [ ! -f $output_csv_file ]; then
            python ${DIR}/ranker_eval/run_evaluations.py \
                    --submit=$result_json_file \
                    --ref=$reference_json_file \
                    --output=$output_csv_file
          fi

          echo $output_csv_file >> $csv_filelist
        done
      fi
    done

    csv_file=${base_dir}/${hash_dir1}/csv-${part}-${c}
    [ ! -f $csv_file ] && cat $(cat $csv_filelist) | sort > $csv_file
    
  done
done

