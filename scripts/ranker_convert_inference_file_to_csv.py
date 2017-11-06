
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input", "", "The input file.")
tf.flags.DEFINE_string("output", "", "The output file.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  assert FLAGS.input, "--input is required"
  assert FLAGS.output, "--output is required"

  with open(FLAGS.input) as F:
    results = json.loads(F.read())
    output_results = []
    for result in results:
      image_id = result['image_id'].encode("utf8")
      for caption in result['captions']:
        caption = caption.encode("utf8")
        output_results.append("\t".join([image_id, caption])+"\n")

  output = open(FLAGS.output, 'w')
  output.writelines(output_results)
  output.close()

if __name__ == "__main__":
  tf.app.run()

