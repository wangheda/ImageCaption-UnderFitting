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
# ==============================================================================
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")

tf.flags.DEFINE_string("inception_checkpoint_file", "",
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_boolean("train_inception", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_boolean("train_inception_with_decay", False,
                        "Whether to train inception submodel variables with decay.")

tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10,
                        "Frequency at which loss and global step are logged.")


# training config
tf.flags.DEFINE_integer("batch_size", 30,
                        "Batch size.")
tf.flags.DEFINE_integer("num_examples_per_epoch", 210000,
                        "Number of examples per epoch of training data.")
tf.flags.DEFINE_string("optimizer", "SGD",
                        "Optimizer for training the model.")
tf.flags.DEFINE_float("initial_learning_rate", 2.0,
                        "Learning rate for the initial phase of training.")
tf.flags.DEFINE_float("learning_rate_decay_factor", 0.5,
                        "Scale learning rate by this factor every num_epochs_per_decay epochs.")
tf.flags.DEFINE_float("num_epochs_per_decay", 8.0,
                        "Scale learning rate by learning_rate_decay_factor every this many epochs.")
tf.flags.DEFINE_float("train_inception_learning_rate", 0.0005,
                        "Learning rate when fine tuning the Inception v3 parameters.")
tf.flags.DEFINE_float("clip_gradients", 5.0,
                        "If not None, clip gradients to this value.")
tf.flags.DEFINE_integer("max_checkpoints_to_keep", 5,
                        "Maximum number of recent checkpoints to preserve.")
tf.flags.DEFINE_float("keep_checkpoint_every_n_hours", 0.25,
                        "Keep a checkpoint every this many hours.")
tf.flags.DEFINE_integer("save_interval_secs", 600,
                        "Save a checkpoint every this many secs.")
tf.flags.DEFINE_string("exclude_variable_patterns", None,
                       "Filter (by comma separated regular expressions) variables that will not be"
                       " loaded from and saved to checkpoints.")

# semantic attention config
tf.flags.DEFINE_boolean("only_attributes_loss", False,
                        "Train only use aux_loss or not.")
tf.flags.DEFINE_string("vocab_file", "",
                       "Text file containing the vocabulary.")
tf.flags.DEFINE_string("attributes_file", "",
                       "Text file containing the concept words.")


import im2txt_model

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.train_dir, "--train_dir is required"

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model = im2txt_model.Im2TxtModel(mode="train")
    model.build()


    # Set up the learning rate.
    learning_rate_decay_fn = None
    if FLAGS.train_inception and not FLAGS.train_inception_with_decay:
      learning_rate = tf.constant(FLAGS.train_inception_learning_rate)
    else:
      learning_rate = tf.constant(FLAGS.initial_learning_rate)
      if FLAGS.learning_rate_decay_factor > 0:
        num_batches_per_epoch = (FLAGS.num_examples_per_epoch /
                                 FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch *
                          FLAGS.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
          return tf.train.exponential_decay(
              learning_rate,
              global_step,
              decay_steps=decay_steps,
              decay_rate=FLAGS.learning_rate_decay_factor,
              staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

    # Set up the training ops.
    if FLAGS.only_attributes_loss:
      loss = model.attributes_loss
    else:
      loss = model.total_loss
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=FLAGS.optimizer,
        clip_gradients=FLAGS.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)


    local_init_op = tf.contrib.slim.learning._USE_DEFAULT

    if FLAGS.exclude_variable_patterns is not None:
      exclude_variables = []
      exclude_variable_names = []

      exclude_variable_patterns = map(lambda x: re.compile(x), FLAGS.exclude_variable_patterns.strip().split(","))
      all_variables = tf.contrib.slim.get_variables()

      for var in all_variables:
        for pattern in exclude_variable_patterns:
          if pattern.match(var.name):
            exclude_variables.append(var)
            exclude_variable_names.append(var.name)
            print("variables to exclude:", var.name)
            break
      print("%d variables to exclude." % len(exclude_variable_names))

      if exclude_variables:
        local_init_op = tf.variables_initializer(exclude_variables)

      variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude_variable_names)

      # Set up the Saver for saving and restoring model checkpoints.
      saver = tf.train.Saver(variables_to_restore,
                             max_to_keep=FLAGS.max_checkpoints_to_keep,
                             keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)
    else:
      # Set up the Saver for saving and restoring model checkpoints.
      saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep,
                             keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

  # Run training.
  tf.contrib.slim.learning.train(
      train_op = train_op,
      logdir = train_dir,
      log_every_n_steps = FLAGS.log_every_n_steps,
      graph = g,
      global_step = model.global_step,
      number_of_steps = FLAGS.number_of_steps,
      local_init_op = local_init_op,
      init_fn = model.init_fn,
      save_summaries_secs = 120,
      save_interval_secs = FLAGS.save_interval_secs,
      saver = saver)


if __name__ == "__main__":
  tf.app.run()
