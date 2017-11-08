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
  def is_pairwise_loss(self):
    raise NotImplementedError()

  def calculate_loss(self, pos_scores, neg_scores, **unused_params):
    raise NotImplementedError()


class HingeLoss(BaseLoss):
  """Hinge loss"""
  def is_pairwise_loss(self):
    return True

  def calculate_loss(self, pos_scores, neg_scores, margin=1.0, **unused_params):
    hinge_loss = tf.maximum(0.0, margin + neg_scores - pos_scores)
    return tf.reduce_mean(hinge_loss)

class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """
  def is_pairwise_loss(self):
    return False

  def calculate_loss(self, predictions, labels, weights=None, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 1e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))
