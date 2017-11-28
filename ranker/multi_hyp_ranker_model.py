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

import losses
import text_models
import match_models
from image_models import image_embedding

from multi_hyp_ranker_readers import get_input_data_tensors

tf.flags.DEFINE_string("text_model", "LstmModel",
                        "The text model.")
tf.flags.DEFINE_string("image_model", "InceptionV3Model",
                        "The image model.")
tf.flags.DEFINE_string("match_model", "MlpModel",
                        "The matching model.")

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
tf.flags.DEFINE_integer("num_readers", 4,
                        "Number of threads for reading data.")
tf.flags.DEFINE_integer("image_height", 299,
                        "Dimensions of Inception v3 input images.")
tf.flags.DEFINE_integer("image_width", 299,
                        "Dimensions of Inception v3 input images.")
tf.flags.DEFINE_integer("image_channel", 3,
                        "Dimensions of Inception v3 input images.")
tf.flags.DEFINE_float("initializer_scale", 0.08,
                        "Scale used to initialize model variables.")

tf.flags.DEFINE_string("loss", "HingeLoss",
                       "Loss.")
tf.flags.DEFINE_float("hinge_loss_margin", 1.0,
                      "Hinge loss margin.")
FLAGS = tf.app.flags.FLAGS


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


class RankerModel(object):
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

    # A float32 Tensor with shape [batch_size, image_model_dim].
    self.image_model_output= None

    # A float32 Tensor with shape [batch_size, lines_per_image, text_model_dim].
    self.text_model_output= None

    # A float32 Tensor with shape [batch_size, lines_per_image].
    self.match_model_output= None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size].
    self.batch_loss = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

    # Global step Tensor.
    self.output_scores = None

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
    print("build inputs!")
    if self.mode == "inference":
      #assert FLAGS.batch_size % FLAGS.lines_per_image == 0
      image_ids, images, captions, seqlens, scores = \
          get_input_data_tensors(data_pattern=FLAGS.input_file_pattern,
                                 is_training=False,
                                 batch_size=FLAGS.batch_size,
                                 num_readers=1)
    else:
      image_ids, images, captions, seqlens, scores = \
          get_input_data_tensors(data_pattern=FLAGS.input_file_pattern,
                                 is_training=True,
                                 batch_size=FLAGS.batch_size,
                                 num_readers=FLAGS.num_readers)

    self.batch_size = FLAGS.batch_size
    batch_size, hyp_num, max_len = captions.shape.as_list()
    self.hyp_num = hyp_num
    self.max_len = max_len

    self.image_ids = image_ids
    self.images = images
    self.captions = captions
    self.seqlens = seqlens
    self.scores = scores
    print(self.images, self.captions, self.seqlens, self.scores)

  def get_image_output(self):
    if self.mode == "inference":
      trainable = False
    else:
      trainable = FLAGS.train_inception or FLAGS.train_inception_with_decay

    inception_output = image_embedding.inception_v3(
        self.images,
        trainable=trainable,
        is_training=self.is_training())

    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # inception_outputs: ([batch_size, feature_size], [batch_size, pos_num, feature_size])
    # tile to ([batch_size * hyp_num, feature_size], [batch_size * hyp_num, pos_num, feature_size])
    if type(inception_output) in [list, tuple]:
      image_model_output, middle_layer = inception_output
      batch_size, feature_size = image_model_output.shape.as_list()
      image_model_output = tf.tile(tf.expand_dims(image_model_output,1), [1, self.hyp_num, 1])
      image_model_output = tf.reshape(image_model_output, shape=[batch_size*self.hyp_num, feature_size])

      batch_size, pos_num, feature_size = middle_layer.shape.as_list()
      middle_layer = tf.tile(tf.expand_dims(middle_layer,1), [1, self.hyp_num, 1, 1])
      middle_layer = tf.reshape(middle_layer, shape=[batch_size*self.hyp_num, pos_num,  feature_size])

      self.image_model_output = (image_model_output, middle_layer)
    else:
      batch_size, feature_size = image_model_output.shape.as_list()
      image_model_output = tf.tile(tf.expand_dims(image_model_output,1), [1, self.hyp_num, 1])
      image_model_output = tf.reshape(image_model_output, shape=[batch_size*self.hyp_num, feature_size])
      self.image_model_output = image_model_output

    print("image output:", self.image_model_output)


  def get_text_output(self):

    text_model_fn = find_class_by_name(FLAGS.text_model, [text_models])
    text_model = text_model_fn()

    batch_size, hyp_num, max_len = self.captions.shape.as_list()

    text_output = text_model.create_model(
          captions = tf.reshape(self.captions, shape=[batch_size * hyp_num, max_len]),
          seqlens = tf.reshape(self.seqlens, shape=[batch_size * hyp_num]),
          global_step = self.global_step,
          initializer = self.initializer,
          mode = self.mode)
    """
    if type(text_output) in [list, tuple]:
      text_output, sequence_layer = text_output
      _, state_size = text_output.shape.as_list()
      text_output = tf.reshape(text_output, shape=[batch_size, hyp_num, state_size])
      _, seq_len, hidden_size = sequence_layer.shape.as_list()
      sequence_layer = tf.reshape(sequence_layer, shape=[batch_size, hyp_num, seq_len, hidden_size])
      self.text_model_output = zip(tf.unstack(text_output, axis=1), tf.unstack(sequence_layer, axis=1))
    else:
      _, state_size = text_output.shape.as_list()
      text_output = tf.reshape(text_output, shape=[batch_size, hyp_num, state_size])
      self.text_model_output = tf.unstack(text_output, axis=1) 
    """
    self.text_model_output = text_output
    print("text output:", self.text_model_output) 

  def build_model(self):

    match_model_fn = find_class_by_name(FLAGS.match_model, [match_models])
    match_model = match_model_fn()

    # model
    outputs =  match_model.create_model(
                  image_input = self.image_model_output, 
                  text_input = self.text_model_output,
                  global_step = self.global_step,
                  initializer = self.initializer,
                  mode = self.mode)
    
    outputs = tf.reshape(outputs, shape=[self.batch_size, self.hyp_num])
    print("outputs:", outputs)
    # loss
    if self.mode == "inference":
      self.output_scores = outputs
    else:
      # Compute losses.
      tf.summary.histogram("losses/output_scores", outputs)
      target_scores = outputs[:,0] # the hyp is sorted
      max_scores = tf.reduce_max(outputs[:,1:], axis=1)
      print(target_scores, max_scores)
      
      loss = tf.maximum(0.0, FLAGS.hinge_loss_margin + max_scores - target_scores)
      batch_loss = tf.reduce_mean(loss)
      tf.losses.add_loss(batch_loss)

      # add auxiliary losses here
      # add auxiliary losses end

      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
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
    self.get_text_output()
    self.build_model()
    self.setup_inception_initializer()
