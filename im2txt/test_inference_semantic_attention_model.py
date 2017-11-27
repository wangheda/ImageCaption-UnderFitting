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
import numpy as np

from inference_utils import vocabulary
from im2txt_model import Im2TxtModel

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("input_file_pattern", "", "The pattern of images.")
tf.flags.DEFINE_float("gpu_memory_fraction", 1.0, "Fraction of gpu memory used in inference.")


tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  assert FLAGS.checkpoint_path, "--checkpoint_path is required"
  assert FLAGS.vocab_file, "--vocab_file is required"
  assert FLAGS.input_file_pattern , "--input_file_pattern is required"
  #assert FLAGS.output, "--output is required"

  # Build the inference graph.
  checkpoint_path = FLAGS.checkpoint_path
  g = tf.Graph()
  with g.as_default():
    tf.logging.info("Building model.")
    model = Im2TxtModel(mode="inference")
    model.build()
    saver = tf.train.Saver()
    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      if not checkpoint_path:
        raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

    def restore_fn(sess):
      tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Successfully loaded checkpoint: %s",
                      os.path.basename(checkpoint_path))
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
    
    t_start = time.time()
    files = tf.gfile.Glob(FLAGS.input_file_pattern)
    print("Found %d images to process" %(len(files)))
    for i, filename in enumerate(files):
      if i > 10:
        break
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()

      print(filename)

      predicted_ids, predicted_ids_lengths, scores, top_n_attributes = sess.run(
        [model.predicted_ids, model.predicted_ids_lengths, model.scores, model.top_n_attributes], 
        feed_dict={"image_feed:0": image})

      predicted_ids = np.transpose(predicted_ids, (0,2,1))   
      scores = np.transpose(scores, (0,2,1))
      attr_probs, attr_ids = top_n_attributes
      attributes = [vocab.id_to_word(w) for w in attr_ids[0]]
      print(" ".join(attributes))
      print(attr_probs[0])
      print(predicted_ids_lengths[0])
      #print(top_n_attributes)
      for caption in predicted_ids[0]:
        print(caption)
        caption = [id for id in caption if id >= 0 and id != FLAGS.end_token]
        sent = [vocab.id_to_word(w) for w in caption]
        print(" ".join(sent))
      """
      captions = generator.beam_search(sess, image)
      image_id = filename.split('.')[0]
      if "/" in image_id:
        image_id = image_id.split("/")[-1]
      result = {}
      result['image_id'] = image_id
      sent = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
      result['caption'] = "".join(sent)
      results.append(result)
      """
  
  t_end = time.time()
  print("time: %f" %(t_end - t_start))
  

if __name__ == "__main__":
  tf.app.run()
