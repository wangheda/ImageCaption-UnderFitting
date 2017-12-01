#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# input
INFERENCE_ALL="${DIR}/../resources/inference_all_v2.list"

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
[ ! -d $ref_dir ] && mkdir -p $ref_dir
[ ! -d ${ref_dir}/TRAIN ] && mkdir -p ${ref_dir}/TRAIN
[ ! -d ${ref_dir}/VALIDATE ] && mkdir -p ${ref_dir}/VALIDATE

for part in VALIDATE TRAIN; do
  if [ $part == "TRAIN" ]; then
    CAPTIONS_FILE=$TRAIN_CAPTIONS_FILE
  else
    CAPTIONS_FILE=$VALIDATE_CAPTIONS_FILE
  fi
  for c in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
    ref_file=${ref_dir}/${part}/reference-${c}.json
    if [ ! -f $ref_file ]; then
      python ${DIR}/ranker_build_reference_file.py \
        --captions_file=$CAPTIONS_FILE \
        --prefix=${c} \
        --output_file=$ref_file
    fi

    csv_filelist=${ref_dir}/${part}/filelist-${part}-${c}
    [ -f $csv_filelist ] && rm $csv_filelist

    for candidate in 0 1 2 3 4; do
      output_file="${ref_dir}/${part}/reference-${c}.json.no-${candidate}"
      if [ ! -f $output_file ]; then
        python ${DIR}/ranker_split_reference_file.py \
              --input=$CAPTIONS_FILE \
              --output=$output_file \
              --prefix=${c} \
              --candidate_id=$candidate
      fi

      result_json_file=$output_file
      output_csv_file="${ref_dir}/${part}/reference-${c}.json.no-${candidate}.csv"

      if [ ! -f $output_csv_file ]; then
        python ${DIR}/ranker_eval/run_evaluations.py \
                --submit=$output_file \
                --ref=$ref_file \
                --output=$output_csv_file
      fi

      echo $output_csv_file >> $csv_filelist
    done

    csv_file=${ref_dir}/refcsv-${part}-${c}
    [ ! -f $csv_file ] && cat $(cat $csv_filelist) | sort > $csv_file
  done
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

        for candidate in 0 1 2; do
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
  output_dir=${DIR}/../data/Ranker_TFRecord_data_v2/${hash_dir1}

  maxlen=30
  strategy1=0
  strategy2=10
  strategy3=0
  num_shards=$SHARD

  for c in $LIST; do
    refcsv_file=${ref_dir}/refcsv-${part}-${c}
    csv_file=${base_dir}/${hash_dir1}/csv-${part}-${c}
    triplets_file=${base_dir}/${hash_dir1}/triplets-${part}-${c}.v2

    if [ ! -f $triplets_file ]; then
      python ${DIR}/build_image_pos_neg_triplets.py \
              --true_captions=$refcsv_file \
              --proposed_captions=$csv_file \
              --output_file=$triplets_file \
              --strategy1=$strategy1 \
              --strategy2=$strategy2 \
              --strategy3=$strategy3
    fi

    marker_file=${base_dir}/${hash_dir1}/marker-${part}-${c}.v2
    if [ ! -f $marker_file ]; then
      CUDA_VISIBLE_DEVICES="" python ${DIR}/build_image_pos_neg_tfrecords.py \
              --word_counts_input_file=${DIR}/../data/word_counts.txt \
              --input_file=$triplets_file \
              --num_shards=$num_shards \
              --maxlen=$maxlen \
              --lines_per_image=$(($strategy1 + $strategy2)) \
              --image_dir=$image_dir \
              --output_prefix="rankertrain-${part}-${c}" \
              --output_dir=$output_dir \
      && touch $marker_file
    fi
  done

done





