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

"""Image embedding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

FLAGS = tf.flags.FLAGS
slim = tf.contrib.slim

def localization_attentions(net, localizations):
  print(net)
  image_shape = net.get_shape().as_list()
  print(image_shape)
  loc_shape = localizations.get_shape().as_list()
  h, w = image_shape[1:3]
  l1 = tf.floor(tf.multiply(localizations[:,:,0], float(w) / float(FLAGS.image_width)))
  u1 = tf.floor(tf.multiply(localizations[:,:,1], float(h) / float(FLAGS.image_height)))
  l2 = tf.ceil(tf.multiply(localizations[:,:,2], float(w) / float(FLAGS.image_width)))
  u2 = tf.ceil(tf.multiply(localizations[:,:,3], float(h) / float(FLAGS.image_height)))
  l1, u1, l2, u2 = map(lambda x: tf.cast(x, dtype=tf.int32), [l1, u1, l2, u2])

  idx_u = tf.reshape(tf.range(0, h, dtype=tf.int32), shape=[h, 1])
  idx_l = tf.reshape(tf.range(0, w, dtype=tf.int32), shape=[1, w])
  masks = []
  for i in xrange(image_shape[0]):
    mask = []
    for j in xrange(loc_shape[1]):
      m = tf.logical_and(tf.logical_and(idx_u >= u1[i,j], idx_u < u2[i,j]),
                         tf.logical_and(idx_l >= l1[i,j], idx_l < l2[i,j]))
      m = tf.cast(m, tf.float32)
      m = tf.multiply(m, 1.0 / (tf.reduce_sum(m) + 1e-9))
      mask.append(m)
    mask = tf.stack(mask, axis=0)
    masks.append(mask)
  masks = tf.stack(masks, axis=0)
  masks = tf.stop_gradient(masks)

  features = tf.einsum("ijkl,imjk->iml", net, masks)
  print(features)
  return features

def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV3",
                 use_box=False,
                 inception_return_tuple=False,
                 localizations=None):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    use_box: Whether to use position information.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images, scope=scope)
        with tf.variable_scope("logits"):
          shape = net.get_shape()
          print(net.get_shape().as_list())
          if inception_return_tuple:
            if FLAGS.localization_attention:
              net = localization_attentions(net, localizations)
              original_net = net
              net = tf.reduce_mean(net, axis=1)
            else:
              original_net = tf.reshape(net, [tf.cast(shape[0],tf.int32), tf.cast(shape[1]*shape[2],tf.int32), tf.cast(shape[3],tf.int32)])
              net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
          elif use_box:
            if FLAGS.localization_attention:
              net = localization_attentions(net, localizations)
            net = tf.reshape(net, [tf.cast(shape[0],tf.int32), tf.cast(shape[1]*shape[2],tf.int32), tf.cast(shape[3],tf.int32)])
          else:
            net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")

          net = slim.dropout(
              net,
              keep_prob=dropout_keep_prob,
              is_training=is_inception_model_training,
              scope="dropout")
          net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  if inception_return_tuple:
    return net, original_net
  else:
    return net
