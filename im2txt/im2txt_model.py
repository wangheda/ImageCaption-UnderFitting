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

import losses
import readers
from train_utils import image_embedding
from train_utils.image_processing import simple_process_image
from train_utils.inputs import get_attributes_target, get_images_and_captions, caption_to_multi_labels

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

tf.flags.DEFINE_string("reader", "OriginalReader",
                        "The originalReader, other options are: LocalizationReader.")
tf.flags.DEFINE_boolean("multiple_references", False,
                        "For RL training, set it to True")
tf.flags.DEFINE_boolean("cropping_images", True,
                       "If you use the localization data, turn it off")

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
tf.flags.DEFINE_integer("image_channel", 3,
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

# semantic attention config
tf.flags.DEFINE_boolean("only_attributes_loss", False,
                        "Train only use aux_loss or not.")
tf.flags.DEFINE_string("vocab_file", "",
                       "Text file containing the vocabulary.")

# reinforcement learning config
tf.flags.DEFINE_boolean("rl_training", False,
                        "Train with reinforcement learning.")
tf.flags.DEFINE_boolean("rl_beam_search_approximation", False,
                        "Whether use beam search to generate sample captions.")
tf.flags.DEFINE_boolean("rl_training_along_with_mle", False,
                        "Train with reinforcement learning with the mle (need to use it along with rl_training).")
tf.flags.DEFINE_string("rl_training_loss", "SelfCriticalLoss",
                        "Type of loss in reinforcement learning.")
tf.flags.DEFINE_integer("max_ref_length", 30, "Max reference length.")


# image config
tf.flags.DEFINE_boolean("l2_normalize_image", False,
                        "Normalize image.")
tf.flags.DEFINE_boolean("localization_attention", False,
                        "Localization attention.")

FLAGS = tf.app.flags.FLAGS


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def get_shape_as_list(tensor):
  return tensor.get_shape().as_list()

def get_rank(tensor):
  return len(get_shape_as_list(tensor))

# padd or truncate the given axis of tensor to max_length
def pad_or_truncate(tensor, lengths, axis, max_length):
  shape = tensor.get_shape().as_list()
  real_shape = tf.shape(tensor)
  target_shape = [l for l in shape]
  target_shape[axis] = max_length

  left_padding, right_padding = [0] * len(shape), [0] * len(shape)
  right_padding[axis] = tf.maximum(max_length - real_shape[axis], 0)
  padded_tensor = tf.pad(tensor, zip(left_padding, right_padding))
  sliced_tensor = tf.slice(padded_tensor, [0] * len(shape), target_shape)
  if lengths is not None:
    clipped_lengths = tf.minimum(lengths, max_length)
  else:
    clipped_lengths = None
  return sliced_tensor, clipped_lengths
  

class Im2TxtModel(object):
  """Image-to-text implementation"""

  def __init__(self, mode):
    """Basic setup.

    Args:
      mode: "train", "eval" or "inference".
    """
    assert mode in ["train", "inference"]
    self.mode = mode

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

    # localization boxes
    self.localizations = None

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

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if FLAGS.reader == "OriginalReader":
      if self.mode == "inference":
        # In inference mode, images and inputs are fed via placeholders.
        image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
        input_feed = tf.placeholder(dtype=tf.int64,
                                    shape=[None],  # batch_size
                                    name="input_feed")

        # Process image and insert batch dimensions.
        images = tf.expand_dims(simple_process_image(image_feed), 0)
        input_seqs = tf.expand_dims(input_feed, 1)

        # No target sequences or input mask in inference mode.
        target_seqs = None
        input_mask = None
      else:
        images, input_seqs, target_seqs, input_mask = get_images_and_captions(is_training=self.is_training)
      if self.mode == "inference":
        target_lengths = None
      else:
        target_lengths = tf.reduce_sum(input_mask, -1)
    elif FLAGS.reader == "ImageCaptionReader":
      reader = readers.ImageCaptionReader(num_refs=FLAGS.num_refs,
                                  max_ref_length=FLAGS.max_ref_length)
      cols = readers.get_input_data_tensors(reader,
                                    data_pattern=FLAGS.input_file_pattern,
                                    batch_size=FLAGS.batch_size,
                                    num_epochs=None,
                                    is_training=True,
                                    num_readers=4)
      if FLAGS.localization_attention:
        images, input_seqs, target_seqs, input_mask, target_lengths, localizations = cols
        self.localizations = localizations
      else:
        images, input_seqs, target_seqs, input_mask, target_lengths = cols
    elif FLAGS.reader == "ImageCaptionTestReader":
      reader = readers.ImageCaptionTestReader()
      cols = readers.get_test_input_data_tensors(reader,
                                    data_pattern=FLAGS.input_file_pattern,
                                    batch_size=FLAGS.batch_size,
                                    num_epochs=1,
                                    num_readers=1)
      if FLAGS.localization_attention:
        images, image_names, localizations = cols
        self.localizations = localizations
      else:
        images, image_names = cols
      self.image_names = image_names
      target_seqs = None
      input_mask = None
      input_seqs = None
      target_lengths = None
        
    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask
    self.target_lengths = target_lengths

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
        inception_return_tuple=FLAGS.inception_return_tuple,
        localizations=self.localizations)
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    if FLAGS.yet_another_inception:
      ya_inception_output = image_embedding.inception_v3(
          self.images,
          trainable=trainable,
          scope="ya_InceptionV3",
          is_training=self.is_training(),
          use_box=FLAGS.use_box,
          inception_return_tuple=FLAGS.inception_return_tuple,
          localizations=self.localizations)
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
          input_mask = self.input_mask,
          target_lengths = self.target_lengths)

    # loss
    if self.mode == "inference":
      if "logits" in outputs:
        tf.nn.softmax(outputs["logits"], name="softmax")
      elif "bs_results" in outputs:
        self.predicted_ids = outputs["bs_results"].predicted_ids
        self.scores = outputs["bs_results"].beam_search_decoder_output.scores
        if "bs_results_lengths" in outputs:
          self.predicted_ids_lengths = outputs["bs_results_lengths"]
      if "top_n_attributes" in outputs:
        self.top_n_attributes = outputs["top_n_attributes"]
    else:
      if "mle_caption_logits" in outputs:
        logits = tf.reshape(outputs["mle_caption_logits"], [-1, FLAGS.vocab_size])
        targets = tf.reshape(self.target_seqs, [-1])
        weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

        # Compute losses.
        mle_loss_fn = losses.SparseSoftmaxCrossEntropyLoss()
        mle_loss = mle_loss_fn.calculate_loss(logits, targets, weights)

        # Logging losses.
        tf.summary.scalar("losses/mle_loss", mle_loss)
        tf.losses.add_loss(mle_loss)

      # caption loss
      if FLAGS.rl_training == True:
        # rl loss
        # load greed caption and sample caption to calculate reward
        target_caption_words = self.target_seqs
        target_caption_lengths = self.target_lengths
        greedy_caption_words = outputs["greedy_caption_words"]
        greedy_caption_lengths = outputs["greedy_caption_lengths"]
        sample_caption_logits = outputs["sample_caption_logits"]
        sample_caption_words = outputs["sample_caption_words"]
        sample_caption_lengths = outputs["sample_caption_lengths"]

        if get_rank(target_caption_words) == 2:
          target_caption_words = tf.expand_dims(target_caption_words, 1)
        if get_rank(target_caption_lengths) == 1:
          target_caption_lengths = tf.expand_dims(target_caption_lengths, 1)

        if get_shape_as_list(target_caption_words)[-1] is None:
          target_caption_words, target_caption_lengths = \
              pad_or_truncate(target_caption_words, target_caption_lengths,
                              axis = -1, max_length = FLAGS.max_ref_length)
        if get_shape_as_list(greedy_caption_words)[-1] is None:
          greedy_caption_words, greedy_caption_lengths = \
              pad_or_truncate(greedy_caption_words, greedy_caption_lengths,
                              axis = -1, max_length = FLAGS.max_caption_length)
        if get_shape_as_list(sample_caption_logits)[1] is None:
          sample_caption_logits, _ = \
              pad_or_truncate(sample_caption_logits, sample_caption_lengths,
                              axis = 1, max_length = FLAGS.max_caption_length)
        if get_shape_as_list(sample_caption_words)[-1] is None:
          sample_caption_words, sample_caption_lengths = \
              pad_or_truncate(sample_caption_words, sample_caption_lengths,
                              axis = -1, max_length = FLAGS.max_caption_length)

        if FLAGS.rl_beam_search_approximation:
          target_caption_words = tf.contrib.seq2seq.tile_batch(target_caption_words, multiplier=FLAGS.beam_width)
          target_caption_lengths = tf.contrib.seq2seq.tile_batch(target_caption_lengths, multiplier=FLAGS.beam_width)
        rl_loss_cls = find_class_by_name(FLAGS.rl_training_loss, [losses])
        rl_loss_fn = rl_loss_cls()
        rl_loss = rl_loss_fn.calculate_loss(
                     target_caption_words   = target_caption_words, 
                     target_caption_lengths = target_caption_lengths, 
                     greedy_caption_words   = greedy_caption_words, 
                     greedy_caption_lengths = greedy_caption_lengths, 
                     sample_caption_words   = sample_caption_words, 
                     sample_caption_lengths = sample_caption_lengths, 
                     sample_caption_logits  = sample_caption_logits
                  )

        tf.losses.add_loss(rl_loss)

      else:
        if "logits" in outputs:
          # prepare logits, targets and weight
          logits = outputs["logits"]
          logits = tf.reshape(logits, [FLAGS.batch_size, -1, FLAGS.vocab_size])
          logits, _ = pad_or_truncate(logits, None, axis=1, max_length=FLAGS.max_ref_length)
          logits = tf.reshape(logits, [-1, FLAGS.vocab_size])
          targets = tf.reshape(self.target_seqs, [-1])
          weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

          # Compute losses.
          loss_fn = losses.SparseSoftmaxCrossEntropyLoss()
          batch_loss = loss_fn.calculate_loss(logits, targets, weights)

          # Logging losses.
          tf.summary.scalar("losses/batch_loss", batch_loss)
          tf.losses.add_loss(batch_loss)

          self.target_cross_entropy_losses = batch_loss # Used in evaluation.
          self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

        # multi-label-loss
        if "attributes_logits" in outputs and "attributes_mask" in outputs:
          attributes_logits = outputs["attributes_logits"]
          attributes_targets = get_attributes_target(self.target_seqs, attributes_mask)
          if FLAGS.use_idf_weighted_attribute_loss:
            attributes_mask = outputs["idf_weighted_mask"]
          else:
            attributes_mask = outputs["attributes_mask"]

          attributes_loss_fn = losses.CrossEntropyLoss()
          attributes_loss = attributes_loss_fn.calculate_loss(
                                    attributes_logits, attributes_targets,
                                    attributes_mask)

          tf.losses.add_loss(attributes_loss)
          tf.summary.scalar("losses/attributes_loss", attributes_loss)
          self.attributes_loss = attributes_loss


        # discriminative loss
        # should be multi-label margin loss, but the loss below is a little different
        if "discriminative_logits" in outputs:
          word_labels = caption_to_multi_labels(self.target_seqs)
          discriminative_loss = tf.losses.hinge_loss(labels=word_labels,
                                                     logits=outputs["discriminative_logits"],
                                                     weights=FLAGS.discriminative_loss_weights)
          tf.summary.scalar("losses/discriminative_loss", discriminative_loss)
          self.discriminative_loss = discriminative_loss


        # word weighted cross entropy loss
        if "word_predictions" in outputs:
          word_loss_fn = losses.CrossEntropyLoss()
          word_loss = word_loss_fn.calculate_loss(outputs["word_predictions"],
                                                  caption_to_multi_labels(self.target_seqs))
          tf.summary.scalar("losses/word_loss", word_loss)
          tf.losses.add_loss(word_loss)
          self.word_loss = word_loss

      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss

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
