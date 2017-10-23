# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# edit by Miao
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import json
import time
import random

import tensorflow as tf

import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_file_pattern", "", "The pattern of images.")
tf.flags.DEFINE_string("output", "", "The output file.")
tf.flags.DEFINE_float("gpu_memory_fraction", 1.0, "Fraction of gpu memory used in inference.")


tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  assert FLAGS.checkpoint_path, "--checkpoint_path is required"
  assert FLAGS.vocab_file, "--vocab_file is required"
  assert FLAGS.input_file_pattern , "--input_file_pattern is required"
  assert FLAGS.output, "--output is required"

  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph(FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  results = []

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  with tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    t_start = time.time()
    files = tf.gfile.Glob(FLAGS.input_file_pattern)
    for i, filename in enumerate(files):
      if i % 100 == 0:
          print(i)
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      image_id = filename.split('.')[0]
      if "/" in image_id:
        image_id = image_id.split("/")[-1]
      result = {}
      result['image_id'] = image_id
      sents = ["".join([vocab.id_to_word(w) for w in caption]) for i, caption in enumerate(captions)]
      sent_ids = [" ".join(map(str, caption)) for caption in captions]
      result['captions'] = sents
      result['caption_ids'] = sent_ids
      results.append(result)
  
  t_end = time.time()
  print("time: %f" %(t_end - t_start))
  output = open(FLAGS.output, 'w')
  json.dump(results, output, ensure_ascii=False, indent=4)
  output.close()

if __name__ == "__main__":
  tf.app.run()
