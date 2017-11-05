# coding: utf-8
# Copyright 2017 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build tfrecord data."""
# __author__ = 'Miao'
# python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import random
import sys
import hashlib

reload(sys)
sys.setdefaultencoding('utf8')
import jieba
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string("input", None,
                       "the caption file")
tf.flags.DEFINE_string("output", None, "The output file.")
tf.flags.DEFINE_string("prefix", None, "The prefix filters image titles.")
tf.flags.DEFINE_integer("candidate_id", None, "The number of candidates.")

FLAGS = tf.flags.FLAGS

def _process_caption_jieba(caption):
    """Processes a Chinese caption string into a list of tonenized words.
    Args:
      caption: A string caption.
    Returns:
      A list of strings; the tokenized caption.
    """
    tokenized_caption = []
    tokenized_caption.extend(jieba.cut(caption, cut_all=False))
    return tokenized_caption


def load_and_process_metadata(captions_file):
    """Loads image metadata from a JSON file and processes the captions.
    Args:
      captions_file: Json file containing caption annotations.
      image_dir: Directory containing the image files.
    Returns:
      A list of ImageMetadata.
    """
    image_id = set([])
    id_to_captions = {}
    with open(captions_file, 'r') as f:
        caption_data = json.load(f)
    for data in caption_data:
        image_name = data['image_id'].split('.')[0]
        descriptions = data['caption']
        if FLAGS.prefix is not None and not image_name.startswith(FLAGS.prefix):
            continue
        if image_name not in image_id:
            id_to_captions.setdefault(image_name, [])
            image_id.add(image_name)
        caption_num = len(descriptions)
        for i in range(caption_num):
            caption_temp = descriptions[i].strip().strip("。").replace('\n', '')
            if caption_temp != '':
                id_to_captions[image_name].append(caption_temp)

    print("Loaded caption metadata for %d images from %s and image_id num is %s" %
          (len(id_to_captions), captions_file, len(image_id)))
    # Process the captions and combine the data into a list of ImageMetadata.
    print("Proccessing captions.")
    image_metadata = []
    image_caption = {}
    num_captions = 0
    id = 0
    for base_filename in image_id:
        image_hash = int(int(hashlib.sha256(base_filename).hexdigest(), 16) % sys.maxint)
        captions = id_to_captions[base_filename]
        if FLAGS.candidate_id >= len(captions):
          caption = captions[-1]
          print("captions abnormal length %d" % len(captions))
        else:
          caption = captions[FLAGS.candidate_id]
        image_caption[base_filename] = caption
    return image_caption

def write_to_json(image_metadata, output_file):
  output_results = []

  for basename, caption in image_metadata.items():
    tmp = {}
    tmp['image_id'] = basename
    tmp['caption'] = caption.encode("utf8")
    output_results.append(tmp)

  output = open(FLAGS.output, 'w')
  json.dump(output_results, output, ensure_ascii=False, indent=4)
  output.close()

if __name__ == "__main__":
  image_metadata = load_and_process_metadata(FLAGS.input)
  write_to_json(image_metadata, FLAGS.output)
  
