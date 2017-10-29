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

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import im2txt_models

from train_utils import image_embedding
from train_utils import image_processing
from train_utils import inputs as input_ops

tf.flags.DEFINE_string("model", "ShowAndTellModel",
                        "The model.")

tf.flags.DEFINE_integer("vocab_size", 12000,
                        "Number of unique words in the vocab (plus 1, for <UNK>)."
                        " The default value is larger than the expected actual vocab size to allow"
                        " for differences between tokenizer versions used in preprocessing. There is"
                        " no harm in using a value greater than the actual vocab size, but using a"
                        " value less than the actual vocab size will result in an error.")
tf.flags.DEFINE_integer("embedding_size", 512,
                        "Word embedding dimension.")

tf.flags.DEFINE_string("image_format", "jpeg",
                        "Image format: jpeg or png.")
tf.flags.DEFINE_integer("values_per_input_shard", 2300,
                        "Approximate number of values per input shard. Used to ensure sufficient"
                        " mixing between shards in training..")
tf.flags.DEFINE_integer("input_queue_capacity_factor", 2,
                        "Minimum number of shards to keep in the input queue.")
tf.flags.DEFINE_integer("num_input_reader_threads", 1,
                        "Number of threads for prefetching SequenceExample protos.")
tf.flags.DEFINE_string("image_feature_name", "image/data",
                        "Name of the SequenceExample context feature containing image data.")
tf.flags.DEFINE_string("caption_feature_name", "image/caption_ids",
                        "Name of the SequenceExample feature list containing integer captions.")
tf.flags.DEFINE_string("flip_caption_feature_name", "image/flip_caption_ids",
                        "Name of the SequenceExample feature list containing integer flip captions.")
tf.flags.DEFINE_integer("num_preprocess_threads", 4,
                        "Number of threads for image preprocessing. Should be a multiple of 2.")
tf.flags.DEFINE_integer("image_height", 299,
                        "Dimensions of Inception v3 input images.")
tf.flags.DEFINE_integer("image_width", 299,
                        "Dimensions of Inception v3 input images.")
tf.flags.DEFINE_float("initializer_scale", 0.08,
                        "Scale used to initialize model variables.")
tf.flags.DEFINE_boolean("support_ingraph", False,
                        "Whether the model supports in-graph inference. If the model supports it, "
                        "the output of the model should contains key 'bs_result'")
tf.flags.DEFINE_boolean("support_flip", False,
                        "Whether the model supports flip image. If the model supports it, "
                        "the SequenceExample should contains feature key 'image/flip_caption_ids'")
tf.flags.DEFINE_boolean("use_box", False,
                        "Whether to remain position information in inception v3 output feature matrix")
tf.flags.DEFINE_boolean("inception_return_tuple", False,
                        "Whether to remain position information in inception v3 output feature matrix, alongside with the origin pooled feature.")
tf.flags.DEFINE_boolean("yet_another_inception", False,
                        "If set true, return two inception output. See image_embedding for details.")

FLAGS = tf.app.flags.FLAGS


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


class Im2TxtModel(object):
  """Image-to-text implementation"""

  def __init__(self, mode):
    """Basic setup.

    Args:
      mode: "train", "eval" or "inference".
    """
    assert mode in ["train", "inference"]
    self.mode = mode

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-FLAGS.initializer_scale,
        maxval=FLAGS.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, image_model_dim].
    self.image_model_output= None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []
    self.ya_inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

    # In-graph inference support
    self.support_ingraph = FLAGS.support_ingraph

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0, flip=False):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=FLAGS.image_height,
                                          width=FLAGS.image_width,
                                          thread_id=thread_id,
                                          image_format=FLAGS.image_format,
                                          flip=flip)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          FLAGS.input_file_pattern,
          is_training=self.is_training(),
          batch_size=FLAGS.batch_size,
          values_per_shard=FLAGS.values_per_input_shard,
          input_queue_capacity_factor=FLAGS.input_queue_capacity_factor,
          num_reader_threads=FLAGS.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert FLAGS.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(FLAGS.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        if FLAGS.support_flip:
          encoded_image, caption, flip_caption = input_ops.parse_sequence_example(
              serialized_sequence_example,
              image_feature=FLAGS.image_feature_name,
              caption_feature=FLAGS.caption_feature_name,
              flip_caption_feature=FLAGS.flip_caption_feature_name)
          # random decides flip or not
          flip_image = self.process_image(encoded_image, thread_id=thread_id, flip=True)
          image = self.process_image(encoded_image, thread_id=thread_id)
          maybe_flip_image, maybe_flip_caption = tf.cond(
            tf.less(tf.random_uniform([],0,1.0), 0.5), 
            lambda: [flip_image, flip_caption], 
            lambda: [image, caption])
          images_and_captions.append([maybe_flip_image, maybe_flip_caption])
        else:
          encoded_image, caption, _ = input_ops.parse_sequence_example(
              serialized_sequence_example,
              image_feature=FLAGS.image_feature_name,
              caption_feature=FLAGS.caption_feature_name)
          image = self.process_image(encoded_image, thread_id=thread_id)
          images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * FLAGS.num_preprocess_threads *
                        FLAGS.batch_size)
      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=FLAGS.batch_size,
                                           queue_capacity=queue_capacity))

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask

  def get_image_output(self):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """
    if self.mode == "inference":
      trainable = False
    else:
      trainable = FLAGS.train_inception or FLAGS.train_inception_with_decay

    inception_output = image_embedding.inception_v3(
        self.images,
        trainable=trainable,
        is_training=self.is_training(),
        use_box=FLAGS.use_box,
        inception_return_tuple=FLAGS.inception_return_tuple)
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    if FLAGS.yet_another_inception:
      ya_inception_output = image_embedding.inception_v3(
          self.images,
          trainable=trainable,
          scope="ya_InceptionV3",
          is_training=self.is_training(),
          use_box=FLAGS.use_box,
          inception_return_tuple=FLAGS.inception_return_tuple)
      self.ya_inception_variables = {v.op.name.lstrip("ya_"): v for v in 
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ya_InceptionV3")}

      if type(inception_output) == tuple:
        inception_output = inception_output + ya_inception_output
      else:
        inception_output = (inception_output, ya_inception_output)

    print(self.inception_variables)
    print(self.ya_inception_variables)

    self.image_model_output = inception_output

  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """

    caption_model_fn = find_class_by_name(FLAGS.model, [im2txt_models])
    caption_model = caption_model_fn()

    # model
    outputs = caption_model.create_model(
          input_seqs = self.input_seqs, 
          image_model_output = self.image_model_output,
          initializer = self.initializer,
          mode = self.mode,
          target_seqs = self.target_seqs,
          global_step = self.global_step,
          input_mask = self.input_mask)

    # loss
    if self.mode == "inference":
      if "logits" in outputs:
        tf.nn.softmax(outputs["logits"], name="softmax")
      elif "bs_results" in outputs:
        self.predicted_ids = outputs["bs_results"].predicted_ids
        self.scores = outputs["bs_results"].beam_search_decoder_output.scores
    else:
      logits = outputs["logits"]
      targets = tf.reshape(self.target_seqs, [-1])
      weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

      # Compute losses.
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                              logits=logits)
      batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                          tf.reduce_sum(weights),
                          name="batch_loss")
      tf.losses.add_loss(batch_loss)

      if "word_predictions" in outputs:
        word_predictions = outputs["word_predictions"]
        word_labels = input_ops.caption_to_multi_labels(self.target_seqs)
        word_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=word_labels,
                                                            logits=word_predictions)
        word_loss = tf.div(tf.reduce_sum(word_loss), 
                           reduce(lambda x,y: x*y, word_loss.get_shape().as_list()),
                           name="word_loss")
        tf.losses.add_loss(word_loss)
        tf.summary.scalar("losses/word_loss", word_loss)
        self.word_loss = word_loss

      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss
      self.target_cross_entropy_losses = losses  # Used in evaluation.
      self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      if FLAGS.yet_another_inception:
        saver = tf.train.Saver(self.inception_variables)
        ya_saver = tf.train.Saver(self.ya_inception_variables)

        def restore_fn(sess):
          tf.logging.info("Restoring Inception variables from checkpoint file %s",
                          FLAGS.inception_checkpoint_file)
          saver.restore(sess, FLAGS.inception_checkpoint_file)
          tf.logging.info("Restoring Inception variables from checkpoint file %s for ya_InceptionV3",
                          FLAGS.inception_checkpoint_file)
          ya_saver.restore(sess, FLAGS.inception_checkpoint_file)

      else:
        saver = tf.train.Saver(self.inception_variables)

        def restore_fn(sess):
          tf.logging.info("Restoring Inception variables from checkpoint file %s",
                          FLAGS.inception_checkpoint_file)
          saver.restore(sess, FLAGS.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.setup_global_step()
    self.build_inputs()
    self.get_image_output()
    self.build_model()
    self.setup_inception_initializer()
