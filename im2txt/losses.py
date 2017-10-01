# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides definitions for non-regularized training or test losses."""

import sys
import numpy as np
import tensorflow as tf
from tensorflow import flags

FLAGS = flags.FLAGS


class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()


class WeightedCrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
     1 -> 0 will be punished hard, while the other way will not punished not hard.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    false_positive_punishment = FLAGS.false_positive_punishment
    false_negative_punishment = FLAGS.false_negative_punishment
    with tf.name_scope("loss_xent_recall"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = false_negative_punishment * float_labels * tf.log(predictions + epsilon) \
          + false_positive_punishment * ( 1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class MeanSquareErrorLoss(BaseLoss):
  """Calculate the MSE loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_mse"):
      shape = predictions.get_shape().as_list()
      dims = range(len(shape))
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      mse_loss = tf.square(float_labels - predictions)
      return tf.reduce_mean(tf.reduce_sum(mse_loss, dims[1:]))


class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, weights=None, **unused_params):
    with tf.name_scope("loss_xent"):
      shape = predictions.get_shape().as_list()
      dims = range(len(shape))
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, dims[1:]))


class IOULoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, weights=None, **unused_params):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
    y_pred (4-D array): (N, H, W, 1)
    y_true (4-D array): (N, H, W, 1)
    Returns:
    float: IOU score
    """
    with tf.name_scope("loss_iou"):
      float_labels = tf.cast(labels, tf.float32)
      print float_labels, predictions
      dims = predictions.get_shape().as_list()[1:]
      if len(dims) == 3:
        H, W, C = dims
        intersection = 2 * tf.reduce_sum(predictions * float_labels, axis=[1,2,3]) + 1e-7
        denominator = tf.reduce_sum(predictions, axis=[1,2,3]) + tf.reduce_sum(float_labels, axis=[1,2,3]) + 1e-7
      elif len(dims) == 2:
        H, W = dims
        intersection = 2 * tf.reduce_sum(predictions * float_labels, axis=[1,2]) + 1e-7
        denominator = tf.reduce_sum(predictions, axis=[1,2]) + tf.reduce_sum(float_labels, axis=[1,2]) + 1e-7
      return - tf.reduce_mean(intersection / denominator)


class IOUCrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, weights=None, **unused_params):
    iou_loss_fn = IOULoss()
    cross_ent_fn = CrossEntropyLoss()
    return 0.0001 * cross_ent_fn.calculate_loss(predictions, labels) + 0.9999 * iou_loss_fn.calculate_loss(predictions, labels)


class HingeLoss(BaseLoss):
  """Calculate the hinge loss between the predictions and labels.

  Note the subgradient is used in the backpropagation, and thus the optimization
  may converge slower. The predictions trained by the hinge loss are between -1
  and +1.
  """

  def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
    with tf.name_scope("loss_hinge"):
      float_labels = tf.cast(labels, tf.float32)
      all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
      all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
      sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
      hinge_loss = tf.maximum(
          all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
      return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))

class SoftmaxLoss(BaseLoss):
  """Calculate the softmax loss between the predictions and labels.

  The function calculates the loss in the following way: first we feed the
  predictions to the softmax activation function and then we calculate
  the minus linear dot product between the logged softmax activations and the
  normalized ground truth label.

  It is an extension to the one-hot label. It allows for more than one positive
  labels for each sample.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_softmax"):
      epsilon = 10e-8
      float_labels = tf.cast(labels, tf.float32)
      # l1 normalization (labels are no less than 0)
      label_rowsum = tf.maximum(
          tf.reduce_sum(float_labels, 1, keep_dims=True),
          epsilon)
      norm_float_labels = tf.div(float_labels, label_rowsum)
      softmax_outputs = tf.nn.softmax(predictions)
      softmax_loss = tf.negative(tf.reduce_sum(
          tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
    return tf.reduce_mean(softmax_loss)

class MultiTaskLoss(BaseLoss):
  """This is a vitural loss
  """
  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    raise NotImplementedError()

  def get_support(self, labels, support_type=None):
    if "," in support_type:
      new_labels = []
      for st in support_type.split(","):
        new_labels.append(tf.cast(self.get_support(labels, st), dtype=tf.float32))
      support_labels = tf.stack(new_labels, axis=3)
      return support_labels
    elif support_type == "label":
      float_labels = tf.cast(labels, dtype=tf.float32)
      return float_labels
    else:
      raise NotImplementedError()

class MultiTaskCrossEntropyLoss(MultiTaskLoss):
  """Calculate the loss between the predictions and labels.
  """
  def calculate_loss(self, predictions, support_predictions, labels, **unused_params):
    support_labels = self.get_support(labels, support_type=FLAGS.support_type)
    ce_loss_fn = CrossEntropyLoss()
    print >> sys.stderr, predictions, labels
    cross_entropy_loss = ce_loss_fn.calculate_loss(predictions, labels, **unused_params)
    cross_entropy_loss2 = ce_loss_fn.calculate_loss(support_predictions, support_labels, **unused_params)
    return cross_entropy_loss * (1.0 - FLAGS.support_loss_percent) + cross_entropy_loss2 * FLAGS.support_loss_percent


class MultiTaskIOULoss(MultiTaskLoss):
  """Calculate the loss between the predictions and labels.
  """
  def calculate_loss(self, predictions, support_predictions, labels, **unused_params):
    support_labels = self.get_support(labels, support_type=FLAGS.support_type)
    iou_loss_fn = IOULoss()
    print >> sys.stderr, predictions, labels
    iou_loss = iou_loss_fn.calculate_loss(predictions, labels, **unused_params)
    iou_loss2 = iou_loss_fn.calculate_loss(support_predictions, support_labels, **unused_params)
    return iou_loss * (1.0 - FLAGS.support_loss_percent) + iou_loss2 * FLAGS.support_loss_percent


class MultiTaskIOUCrossEntropyLoss(MultiTaskLoss):
  """Calculate the loss between the predictions and labels.
  """
  def calculate_loss(self, predictions, support_predictions, labels, **unused_params):
    support_labels = self.get_support(labels, support_type=FLAGS.support_type)
    iou_loss_fn = IOUCrossEntropyLoss()
    print >> sys.stderr, predictions, labels
    iou_loss = iou_loss_fn.calculate_loss(predictions, labels, **unused_params)
    iou_loss2 = iou_loss_fn.calculate_loss(support_predictions, support_labels, **unused_params)
    return iou_loss * (1.0 - FLAGS.support_loss_percent) + iou_loss2 * FLAGS.support_loss_percent


