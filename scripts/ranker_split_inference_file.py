
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input", "", "The input file.")
tf.flags.DEFINE_string("output", "", "The output file.")
tf.flags.DEFINE_integer("candidate_id", None, "The number of candidates.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  assert FLAGS.input, "--input is required"
  assert FLAGS.output, "--output is required"
  assert FLAGS.candidate_id is not None, "--candidate_id is required"

  with open(FLAGS.input) as F:
    results = json.loads(F.read())
    output_results = []
    for result in results:
      tmp = {}
      tmp['image_id'] = result['image_id']
      tmp['caption'] = result['captions'][FLAGS.candidate_id].encode("utf8")
      output_results.append(tmp)

  output = open(FLAGS.output, 'w')
  json.dump(output_results, output, ensure_ascii=False, indent=4)
  output.close()

if __name__ == "__main__":
  tf.app.run()
