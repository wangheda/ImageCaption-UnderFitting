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
import ranker_model
import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_file_pattern", "", "The pattern of images.")
tf.flags.DEFINE_string("output", "", "The output file.")

tf.flags.DEFINE_integer("batch_size", 200,
                        "Batch size.")
tf.flags.DEFINE_integer("num_steps", 200,
                        "Number of steps(temporary).")

tf.logging.set_verbosity(tf.logging.INFO)

def _create_restore_fn(checkpoint_path, saver):
  if tf.gfile.IsDirectory(checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    if not checkpoint_path:
      raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

  def _restore_fn(sess):
    tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
    saver.restore(sess, checkpoint_path)
    tf.logging.info("Successfully loaded checkpoint: %s",
                    os.path.basename(checkpoint_path))

  return _restore_fn


def main(_):
  assert FLAGS.checkpoint_path, "--checkpoint_path is required"
  assert FLAGS.vocab_file, "--vocab_file is required"
  assert FLAGS.input_file_pattern , "--input_file_pattern is required"
  assert FLAGS.output, "--output is required"

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = ranker_model.RankerModel(mode="inference")
    model.build()
    saver = tf.train.Saver()
    restore_fn = _create_restore_fn(FLAGS.checkpoint_path, saver)
  g.finalize()

  results = []
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # available beam search parameters.
    t_start = time.time()
    tf.train.start_queue_runners()

    k = 1
    try:
      for i in xrange(FLAGS.num_steps):
        if k % 100 == 0:
          print(k)
        k += 1

        image_ids, captions, seqlens, scores = sess.run([model.image_ids, model.captions, model.seqlens, model.output_scores])

        for i in xrange(len(image_ids)):
          image_id = image_ids[i]
          caption = captions[i,:]
          seqlen = seqlens[i]
          score = scores[i,0]
          caption_str = "".join([vocab.id_to_word(w) for w in caption[:seqlen]])
          results.append("\t".join([image_id, caption_str, str(score)])+"\n")
    except Exception as e:
      print(e)
    finally:
      pass

  t_end = time.time()
  print("time: %f" %(t_end - t_start))
  output = open(FLAGS.output, 'w')
  output.writelines(results)
  output.close()

if __name__ == "__main__":
  tf.app.run()
