#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# model directory
model_dir="${DIR}/../model/show_and_tell_advanced_model_attention_finetune_with_decay_da/"

# checkpoints
ckpt_lists="model.ckpt-561157, model.ckpt-581505, model.ckpt-621091, model.ckpt-640713, model.ckpt-680637"

# output
output_path="${model_dir}/model.ckpt-avg"

CUDA_VISIBLE_DEVICES=0 python ${DIR}/avg_checkpoints.py \
  --checkpoints=${ckpt_lists} \
  --prefix=${model_dir} \
  --output_path=${output_path}

